"""B.17: dump the predictive distribution of a v2g / v2m2 run on the frozen TEST rows.

Runs AFTER eval_seq.py produced `comparison_errors_test.npz` for the arm. For every scored
row (per target) it writes, in PHYSICAL units:

  pi (n, K), mu (n, K), sigma (n, K)   the arm's predictive mixture (K = 1 for v2g)
  y, se_model, dt_ms, shot, row, is_peak   copied from the arm's npz (row identity)
  backbone_pred                          the frozen seq_v2 control's point prediction
  energy                                 log1p(mean fast-channel increment^2) at the row
                                         (the sec. 8bb p_obs proxy, the B.16 gate input)

Two pairing guards abort on any drift: the arm's mean re-derived here must reproduce the
npz `se_model` of every row, and the backbone lookup must reproduce the control npz's
`se_model` likewise. Output: `dist_test.npz` next to the arm's artifacts.

Env: the arm's eval env (CES_OUTPUT_DIR, CES_SPLIT_DIR, CES_CONTROL_METRICS, CES_SEQ_MODEL,
CES_PER_SHOT_NORM, ...) plus CES_B17_CONTROL_DIR = the frozen backbone run dir.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SEQ = HERE.parent / "seq"
sys.path.insert(0, str(SEQ))
sys.path.insert(1, str(HERE.parents[1]))  # ces_prediction/

from evaluate import _load_stats, build_clean_val_subset  # noqa: E402
from seq_data import load_grid_files, build_blocks, N_FAST_CHANNELS  # noqa: E402
from seq_models import SEQ_MODELS  # noqa: E402

TARGETS = ("CES_TI", "CES_VT")


def lookup_tables(model, grid, dims, stats, names, per_shot, device, want_dist):
    """name -> {time -> (mean_phys (2,), pi (2,K), mu_phys (2,K), sigma_phys (2,K), energy)}."""
    t_mean = np.asarray(stats["target"]["mean"], dtype=np.float64)
    t_std = np.asarray(stats["target"]["std"], dtype=np.float64)
    out = {}
    with torch.no_grad():
        for name in names:
            if name not in grid:
                continue
            table = {}
            for blk in build_blocks(grid[name], dims, stats, per_shot_norm=per_shot):
                x = torch.from_numpy(blk["x"]).unsqueeze(0).to(device)
                fast = x[0, :, :N_FAST_CHANNELS]
                d = torch.zeros_like(fast)
                d[1:] = fast[1:] - fast[:-1]
                energy = torch.log1p(d.pow(2).mean(dim=-1)).cpu().numpy()
                if want_dist:
                    params = model(x, None, return_dist=True)[0]        # (L, 2, P)
                    pi, mu, sigma = model.decode(params)
                    mean = model.mean(params).cpu().numpy().astype(np.float64)
                    pi, mu, sigma = (v.cpu().numpy().astype(np.float64) for v in (pi, mu, sigma))
                    mu_phys = mu * t_std[None, :, None] + t_mean[None, :, None]
                    sigma_phys = sigma * t_std[None, :, None]
                else:
                    mean = model(x)[0].cpu().numpy().astype(np.float64)
                    pi = mu_phys = sigma_phys = None
                mean_phys = mean * t_std + t_mean
                for i, t in enumerate(blk["time"]):
                    key = float(t)
                    if key in table:
                        raise SystemExit(f"FATAL: duplicate time {key} in {name}")
                    table[key] = (mean_phys[i],
                                  None if pi is None else pi[i],
                                  None if pi is None else mu_phys[i],
                                  None if pi is None else sigma_phys[i],
                                  float(energy[i]))
            out[name] = table
    return out


def main():
    root = HERE.parents[2]
    data_dir = Path(os.getenv("CES_DATA_DIR", root / "data"))
    out_dir = Path(os.environ["CES_OUTPUT_DIR"])
    split_dir = Path(os.environ["CES_SPLIT_DIR"])
    control_dir = Path(os.environ["CES_B17_CONTROL_DIR"])
    variant = os.environ["CES_SEQ_MODEL"]
    per_shot = os.getenv("CES_PER_SHOT_NORM", "0") == "1"
    drop_stuck = os.getenv("CES_DROP_STUCK_TARGETS", "1") == "1"
    device = torch.device(os.getenv("CES_SEQ_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))

    arm_metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    if arm_metrics.get("dist") not in ("gauss", "mix2"):
        raise SystemExit(f"FATAL: {out_dir} is not a distribution-head run")
    ctrl_metrics = json.loads((control_dir / "metrics.json").read_text(encoding="utf-8"))
    if ctrl_metrics.get("seq_model") != "v2" or bool(ctrl_metrics.get("per_shot_input_norm")) != per_shot:
        raise SystemExit("FATAL: control dir is not a seq_v2 run with the same input treatment")
    harness_stats = _load_stats(json.loads(Path(os.environ["CES_CONTROL_METRICS"]).read_text(encoding="utf-8")))
    manifest = json.loads((split_dir / "split_manifest.json").read_text(encoding="utf-8"))
    eval_files = set(manifest["test_files"])

    grid, dims = load_grid_files(data_dir, drop_stuck)
    arm = SEQ_MODELS[variant]().to(device)
    arm.load_state_dict(torch.load(out_dir / "weights" / "seq_lstm.pth", map_location="cpu"))
    arm.eval()
    ctrl = SEQ_MODELS["v2"]().to(device)
    ctrl.load_state_dict(torch.load(control_dir / "weights" / "seq_lstm.pth", map_location="cpu"))
    ctrl.eval()
    names = sorted(eval_files)
    tab_arm = lookup_tables(arm, grid, dims, _load_stats(arm_metrics), names, per_shot, device, True)
    tab_ctrl = lookup_tables(ctrl, grid, dims, _load_stats(ctrl_metrics), names, per_shot, device, False)

    # the harness population, for (shot, row) -> time
    window_size = int(os.getenv("CES_WINDOW_SIZE", "4"))
    seed = int(os.getenv("CES_SEED", "42"))
    max_val = int(os.getenv("CES_MAX_VAL_SAMPLES", "40000"))
    dataset, _, _ = build_clean_val_subset(data_dir, window_size, harness_stats, eval_files, max_val, seed)
    time_col = int(dataset._column_slices["time"])
    files = [Path(p).name for p in dataset.valid_files]

    z_arm = np.load(out_dir / "comparison_errors_test.npz")
    z_ctrl = np.load(control_dir / "comparison_errors_test.npz")
    K = arm.n_comp
    out = {}
    for ti, t in enumerate(TARGETS):
        shot, row = z_arm[f"{t}_shot"], z_arm[f"{t}_row"]
        y = z_arm[f"{t}_y_true"].astype(np.float64)
        # the frozen B.1 / B.5 control npz predate the `_row` key (2026-08-21): rows are
        # pinned by (shot, dt, y_true), the same identity rescore_b13 relies on
        same_rows = (np.array_equal(shot, z_ctrl[f"{t}_shot"])
                     and np.array_equal(z_arm[f"{t}_dt_ms"], z_ctrl[f"{t}_dt_ms"], equal_nan=True)
                     and np.array_equal(y, z_ctrl[f"{t}_y_true"].astype(np.float64)))
        if not same_rows:
            if not os.getenv("CES_SEQ_MAX_FILES"):
                raise SystemExit(f"FATAL: {t}: arm and control npz are not the same rows")
            # smoke only: the arm scored the SMOKE_OVERRIDES population, the frozen control
            # the full one -- pairing with the control npz is impossible by construction
            print(f"[b17] {t}: smoke population differs from the control npz; control guard skipped")
        n = len(y)
        pi = np.zeros((n, K)); mu = np.zeros((n, K)); sg = np.zeros((n, K))
        bb = np.zeros(n); en = np.zeros(n)
        max_dev_arm = max_dev_ctrl = 0.0
        for i in range(n):
            name = files[int(shot[i])]
            key = float(dataset.file_arrays[int(shot[i])][int(row[i]), time_col])
            m_a, p_a, mu_a, sg_a, e_a = tab_arm[name][key]
            m_c = tab_ctrl[name][key][0]
            pi[i], mu[i], sg[i], en[i], bb[i] = p_a[ti], mu_a[ti], sg_a[ti], e_a, m_c[ti]
            max_dev_arm = max(max_dev_arm, abs((m_a[ti] - y[i]) ** 2 - float(z_arm[f"{t}_se_model"][i])))
            if same_rows:
                max_dev_ctrl = max(max_dev_ctrl, abs((m_c[ti] - y[i]) ** 2 - float(z_ctrl[f"{t}_se_model"][i])))
        scale = float(np.mean(z_arm[f"{t}_se_model"]))
        # float32 lookup vs float64 here: agree to ~1e-4 relative of the mean SE
        for lbl, dev in (("arm", max_dev_arm), ("control", max_dev_ctrl)):
            if dev > 1e-3 * scale:
                raise SystemExit(f"FATAL: {t}: {lbl} mean does not reproduce its npz se_model "
                                 f"(max dev {dev:.4g} vs mean SE {scale:.4g})")
        print(f"[b17] {t}: n={n} K={K} pairing guards ok (arm dev {max_dev_arm:.3g}, control dev {max_dev_ctrl:.3g})")
        out.update({f"{t}_pi": pi, f"{t}_mu": mu, f"{t}_sigma": sg, f"{t}_y": y, f"{t}_backbone_pred": bb,
                    f"{t}_energy": en, f"{t}_dt_ms": z_arm[f"{t}_dt_ms"], f"{t}_shot": shot, f"{t}_row": row,
                    f"{t}_is_peak": z_arm[f"{t}_is_peak"], f"{t}_se_model": z_arm[f"{t}_se_model"],
                    f"{t}_se_control": z_ctrl[f"{t}_se_model"] if same_rows else (bb - y) ** 2})
    np.savez(out_dir / "dist_test.npz", **out)
    print(f"[b17] wrote {out_dir / 'dist_test.npz'}", flush=True)


if __name__ == "__main__":
    main()
    os._exit(0)
