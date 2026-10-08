"""Train one B.12 arm on one split and score its fixed TEST hidden points.

Usage (from repo root):
  py ces_prediction/experiments/b12/train_b12.py --arm np --seed 42 --cut 3000 --out data/.b12_np_cut3000_s42

Same for every arm (PREREGISTRATION_B12.md §4): split manifest (data/.b1_manifest_s{seed}),
held-free data, fixed val/test masks, fresh train masks each epoch, loss = per-target
masked MSE on HIDDEN observed rows only, AdamW, ReduceLROnPlateau, early stop on val
hidden-row MSE (patience 6, cap 40 epochs), best-val weights kept. Arms with a
`window` attribute train on random W-step crops and are scored with sliding windows.

Output: weights.pth, metrics.json, test_points.npz (shot, row, target, truth, pred in
physical units), val_points.npz.
"""

import argparse
import json
import random
import sys
import time as _time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from b12_data import prepare, file_features, draw_hidden  # noqa: E402
from b12_models import build  # noqa: E402

REPO = HERE.parents[2]
DATA = REPO / "data"
MASK_DIR = DATA / ".b12_masks"
TOKAMIND_DIR = DATA / ".b12_tokamind"
MAX_EPOCHS, PATIENCE, BATCH = 40, 6, 16


def file_items(prep, names, masks):
    """name -> list of block dicts, with the file's concatenated features attached."""
    items = []
    for n in names:
        blocks = file_features(prep["grid"][n], masks[n], prep["stats"])
        shot_x = np.concatenate([b["x"] for b in blocks])
        for b in blocks:
            b["name"], b["shot_x"] = n, shot_x
            items.append(b)
    return items


def batches(items, device, shuffle, rng, crop=None):
    order = list(range(len(items)))
    if shuffle:
        rng.shuffle(order)
    for s in range(0, len(order), BATCH):
        idx = order[s:s + BATCH]
        segs = []
        for i in idx:
            b = items[i]
            n = b["x"].shape[0]
            a = rng.randrange(0, n - crop + 1) if (crop and n > crop) else 0
            e = a + crop if (crop and n > crop) else n
            segs.append((b, a, e))
        L = max(e - a for _, a, e in segs)
        S = max(b["shot_x"].shape[0] for b, _, _ in segs)
        B = len(segs)
        x = torch.zeros(B, L, segs[0][0]["x"].shape[1])
        y, hid = torch.zeros(B, L, 2), torch.zeros(B, L, 2)
        shot_x = torch.zeros(B, S, x.shape[2])
        lengths, shot_len = torch.zeros(B, dtype=torch.long), torch.zeros(B, dtype=torch.long)
        for j, (b, a, e) in enumerate(segs):
            x[j, :e - a] = torch.from_numpy(b["x"][a:e])
            y[j, :e - a] = torch.from_numpy(b["y"][a:e])
            hid[j, :e - a] = torch.from_numpy(b["hid"][a:e])
            lengths[j] = e - a
            shot_x[j, :b["shot_x"].shape[0]] = torch.from_numpy(b["shot_x"])
            shot_len[j] = b["shot_x"].shape[0]
        yield segs, x.to(device), y.to(device), hid.to(device), lengths, shot_x.to(device), shot_len


def run_pass(model, items, device, optimizer=None, rng=None, crop=None):
    training = optimizer is not None
    model.train(training)
    se_sum, n = torch.zeros(2), torch.zeros(2)
    for _, x, y, hid, lengths, shot_x, shot_len in batches(items, device, training, rng or random.Random(0),
                                                           crop if training else None):
        with torch.set_grad_enabled(training):
            out = model(x, lengths, shot_x, shot_len)
            se = ((out - y) ** 2) * hid
            per_t = se.sum((0, 1)) / hid.sum((0, 1)).clamp(min=1.0)
            loss = per_t.sum()
            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        se_sum += se.detach().sum((0, 1)).cpu()
        n += hid.sum((0, 1)).cpu()
    per_t = (se_sum / n.clamp(min=1.0)).numpy()
    return float(per_t.sum()), per_t.tolist()


@torch.no_grad()
def predict_points(model, items, device, stats):
    """Hidden observed rows -> structured arrays in physical units."""
    model.eval()
    mean, std = stats["target"]["mean"], stats["target"]["std"]
    rows = {"shot": [], "row": [], "target": [], "truth": [], "pred": []}
    for segs, x, y, hid, lengths, shot_x, shot_len in batches(items, device, False, random.Random(0)):
        out = model(x, lengths, shot_x, shot_len).cpu().numpy()
        for j, (b, a, e) in enumerate(segs):
            for t in range(2):
                r = np.flatnonzero(b["hid"][a:e, t] > 0)
                rows["shot"] += [b["name"]] * len(r)
                rows["row"] += (b["row0"] + a + r).tolist()
                rows["target"] += [t] * len(r)
                rows["truth"] += (b["y"][a + r, t] * std[t] + mean[t]).tolist()
                rows["pred"] += (out[j, r, t] * std[t] + mean[t]).tolist()
    return {k: np.asarray(v) for k, v in rows.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--cut", type=float, required=True, help="CES_TI spike cut [eV]; 0 = inclusive")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-files", type=int, default=0, help="smoke only")
    ap.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device(args.device)

    t0 = _time.time()
    prep = prepare(DATA, DATA / f".b1_manifest_s{args.seed}", args.cut,
                   MASK_DIR / ("smoke" if args.max_files else ""), args.seed)
    train, val, test = prep["train"], prep["val"], prep["test"]
    if args.max_files:
        train, val, test = train[:args.max_files], val[:4], test[:4]
    val_items = file_items(prep, val, prep["val_masks"])
    test_items = file_items(prep, test, prep["test_masks"])

    model = build(args.arm, TOKAMIND_DIR).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    crop = getattr(model, "window", None)
    if hasattr(model, "pretrained_parameters"):
        # Same two-rate schedule for tm_pre and tm_scratch: the backbone-side parameters
        # get the lower rate in both, so initial weights stay the only difference.
        bb = {id(p) for p in model.pretrained_parameters()}
        groups = [{"params": [p for p in model.parameters() if id(p) in bb], "lr": 3e-4},
                  {"params": [p for p in model.parameters() if id(p) not in bb], "lr": 1e-3}]
    else:
        groups = [{"params": list(model.parameters()), "lr": 1e-3}]
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", patience=2, factor=0.5)
    print(f"[b12] arm={args.arm} seed={args.seed} cut={args.cut:g} params={n_params:,} "
          f"files train/val/test={len(train)}/{len(val)}/{len(test)} device={device.type} "
          f"({_time.time() - t0:.0f}s prep)", flush=True)

    best, best_state, best_epoch, history = float("inf"), None, -1, []
    for epoch in range(args.epochs):
        masks = {n: draw_hidden(prep["grid"][n], prep["runs"], np.random.default_rng([args.seed, epoch, i]))
                 for i, n in enumerate(train)}
        tr_items = file_items(prep, train, masks)
        tr, tr_t = run_pass(model, tr_items, device, opt, rng, crop)
        va, va_t = run_pass(model, val_items, device)
        sched.step(va)
        star = ""
        if va < best:
            best, best_epoch = va, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            star = " *"
        history.append({"epoch": epoch + 1, "train": tr_t, "val": va_t})
        print(f"[b12] epoch {epoch + 1:02d} train={tr:.4f} val={va:.4f} "
              f"(TI {va_t[0]:.4f} VT {va_t[1]:.4f}){star} {_time.time() - t0:.0f}s", flush=True)
        if epoch - best_epoch >= PATIENCE:
            break

    model.load_state_dict(best_state)
    torch.save(best_state, out / "weights.pth")
    sets = [("val", val_items), ("test", test_items)]
    sets += [(f"test_sparse{k}", file_items(prep, test, m)) for k, m in prep["sparse_masks"].items()]
    for split, items in sets:
        np.savez_compressed(out / f"{split}_points.npz", **predict_points(model, items, device, prep["stats"]))
    (out / "metrics.json").write_text(json.dumps({
        "arm": args.arm, "seed": args.seed, "ti_spike_cut_ev": args.cut, "params": n_params,
        "best_epoch": best_epoch + 1, "best_val_hidden_mse_sum": best, "history": history,
        "smoke": bool(args.max_files), "minutes": (_time.time() - t0) / 60,
    }, indent=2), encoding="utf-8")
    print(f"[b12] done best_epoch={best_epoch + 1} val={best:.4f} -> {out}")


if __name__ == "__main__":
    main()
