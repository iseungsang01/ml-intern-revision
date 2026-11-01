"""B.25 (PREREGISTRATION_B25.md): per-row readout of frozen v4k checkpoints -- inference only.

For each (pop, seed) and arm this runs ONE whole-block forward of the checkpoint (the eval_seq
inference path) over the split's files and stores, for every observed row whose block has an
earlier observation of the same target: target y, prediction yhat, anchor a, anchor sd s, gain g
(k for the fusion arm, 1 for additive), previous observed value, raw-unit T_i, file index.
a and s come from re-calling the model's two Kalman cells on the same input; they are checked bit
for bit against the model's own anchor_only path (FATAL otherwise).

  part A (TEST): .b16_v4kjtf (B.21 fusion) and .b16_v4kjt (B.19 additive)
  part B (VAL):  .b16_v4kjt (B.19, val-MSE selected) and .b20_v4kjt (B.20, val-median-AE selected)

Output: data/.b25_rows.npz (all values in physical units).
Usage (repo root):  py ces_prediction/experiments/b16_physics/residual_rows_b25.py
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_b16 import DATA, POPS, SEEDS, SEQ_DIR, base_env  # noqa: E402

sys.path.insert(0, str(SEQ_DIR))
import torch  # noqa: E402
from seq_data import build_blocks, fit_stats, load_grid_files  # noqa: E402
from seq_models import SEQ_MODELS  # noqa: E402

PARTS = {"A": ("test", (("fusion", ".b16_v4kjtf", "v4kjtf"), ("additive", ".b16_v4kjt", "v4kjt"))),
         "B": ("val", (("mse_sel", ".b16_v4kjt", "v4kjt"), ("medae_sel", ".b20_v4kjt", "v4kjt")))}
FIELDS = ("y", "yhat", "a", "s", "g", "prev", "file")


def readout(model, blocks, device, mean, std):
    rows = {t: {f: [] for f in FIELDS} for t in range(2)}
    with torch.no_grad():
        for fi, blk in blocks:
            x = torch.from_numpy(blk["x"]).unsqueeze(0).to(device)
            obs = torch.from_numpy(blk["mask"]).unsqueeze(0).to(device)
            model.anchor_only = False
            yhat = model(x, None, obs)[0].cpu().numpy()
            g = torch.stack(model.last_gate, dim=-1)[0].cpu().numpy()
            xin = x[..., :model.n_in]
            obs_prev = torch.zeros_like(obs)
            obs_prev[:, 1:, :] = obs[:, :-1, :]
            cells = (model.kalman_ti, model.kalman_vt)
            a_s = [cells[t](xin[..., model.col_carried[t]], obs_prev[..., t]) for t in range(2)]
            a = torch.stack([v[0] for v in a_s], dim=-1)[0]
            s = torch.stack([v[1] for v in a_s], dim=-1)[0]
            model.anchor_only = True
            a_ref = model(x, None, obs)[0]
            model.anchor_only = False
            if not torch.equal(a, a_ref):
                raise SystemExit("FATAL: re-called Kalman anchor differs from the model's anchor_only path")
            a, s = a.cpu().numpy(), s.cpu().numpy()
            if not model.gated and not model.fusion:
                g = np.ones_like(g)
            y, m = blk["y"], blk["mask"]
            for t in range(2):
                last = np.nan
                for i in range(len(y)):
                    if m[i, t] > 0.5:
                        if np.isfinite(last):
                            r = rows[t]
                            r["y"].append(y[i, t]); r["yhat"].append(yhat[i, t]); r["a"].append(a[i, t])
                            r["s"].append(s[i, t]); r["g"].append(g[i, t]); r["prev"].append(last)
                            r["file"].append(fi)
                        last = y[i, t]
    out = {}
    for t in range(2):
        arr = {f: np.asarray(v, dtype=np.float64) for f, v in rows[t].items()}
        for f in ("y", "yhat", "a", "prev"):
            arr[f] = arr[f] * std[t] + mean[t]
        arr["s"] = arr["s"] * std[t]
        out[t] = arr
    return out


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("FATAL: no GPU visible -- B.25 is a GPU inference batch (run unsandboxed)")
    save = {}
    for pop, p in POPS.items():
        env = base_env(pop, "v4kjt", False)
        for k in ("CES_TI_SPIKE_CUT_EV", "CES_DROP_STUCK_TARGETS", "CES_PER_SHOT_NORM"):
            os.environ[k] = env[k]
        cut = float(env["CES_TI_SPIKE_CUT_EV"])
        grid, dims = load_grid_files(DATA, env["CES_DROP_STUCK_TARGETS"] == "1", cut)
        for s in SEEDS:
            man = json.loads((DATA / p["split"].format(s=s) / "split_manifest.json").read_text(encoding="utf-8"))
            train = [n for n in man["train_files"] if n in grid]
            stats = fit_stats(grid, dims, train)
            mean = np.asarray(stats["target"]["mean"], dtype=np.float64)
            std = np.asarray(stats["target"]["std"], dtype=np.float64)
            for part, (split_tag, arms) in PARTS.items():
                names = sorted(n for n in man[f"{split_tag}_files"] if n in grid)
                blocks = [(fi, b) for fi, n in enumerate(names)
                          for b in build_blocks(grid[n], dims, stats, per_shot_norm=True)]
                for arm, prefix, variant in arms:
                    d = DATA / f"{prefix}_{pop}_s{s}"
                    met = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
                    ck = met["normalization"]["stats"]["target"]
                    if not (np.allclose(ck["mean"], mean, rtol=0, atol=1e-6 * np.abs(mean).max())
                            and np.allclose(ck["std"], std, rtol=1e-6)):
                        raise SystemExit(f"FATAL: target stats of {d.name} differ from the recomputed TRAIN stats")
                    model = SEQ_MODELS[variant]().to(device)
                    model.load_state_dict(torch.load(d / "weights" / "seq_lstm.pth", map_location="cpu"))
                    model.eval()
                    rows = readout(model, blocks, device, mean, std)
                    for t, arr in rows.items():
                        for f, v in arr.items():
                            save[f"{part}|{pop}|{s}|{arm}|{t}|{f}"] = v
                    print(f"[b25] {part} {pop} s{s} {arm:9s} rows T_i {len(rows[0]['y'])} V_rot {len(rows[1]['y'])}",
                          flush=True)
    np.savez_compressed(DATA / ".b25_rows.npz", **save)
    print(f"wrote {DATA / '.b25_rows.npz'}")


if __name__ == "__main__":
    main()
