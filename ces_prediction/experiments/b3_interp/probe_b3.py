"""B.3 pre-registered measurements on the confirmed runs (PREREGISTRATION_W2.md sec. 6, B.3).

For each confirmatory run (data/.b3c_{variant}_s{seed}):

  (1) latent linear probes -- fit OLS on TRAIN-file grid steps (seeded subsample),
      report R^2 on TEST-file grid steps, full-latent and per-dimension, against the
      four pre-registered physical quantities:
        te_mean     mean of the 4 ECEI channels' (per-shot) z-values at the step
                    (electron-temperature proxy -- ECEI measures T_e)
        activity    mean over the 9 BES channels of the trailing-100 ms (10-step)
                    causal std of their z-values (local fluctuation activity)
        staleness   log1p(seconds since the last CES_TI observation)  [input ch]
        carried     last observed CES_TI (global target z)            [input ch]
      plus the slow-side pair (staleness/carried of CES_VT) probed from z_vt.
  (2) exact term decomposition stats on TEST steps: per-latent-dim contribution
      w_k * z_k (mean |.|, std), anchor std, correction/total share.
  (3) structural routing check: perturbing the 15 fast channels leaves the V_rot
      output BIT-IDENTICAL while CES_TI responds (CPU, single forward pair).

Usage (repo root):
  py ces_prediction/experiments/b3_interp/probe_b3.py --variant b3k8
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
CES_DIR = REPO_ROOT / "ces_prediction"
SEQ_DIR = CES_DIR / "experiments" / "seq"
DATA = REPO_ROOT / "data"
sys.path.insert(0, str(SEQ_DIR))
sys.path.insert(1, str(CES_DIR))

from evaluate import _load_stats  # noqa: E402
from seq_data import load_grid_files, build_blocks, N_FAST_CHANNELS  # noqa: E402
from train_seq import SEQ_MODELS  # noqa: E402

SEEDS = (42, 1, 7, 123)
TI_SPIKE_CUT_EV = 3000.0     # confirmed protocol -- pinned, never inherited
DROP_STUCK = True
PER_SHOT = True              # seq-family definition
MAX_FIT_STEPS = 200_000
MAX_EVAL_STEPS = 200_000
PROBE_SEED = 12345

# BES: fast channels 0..8, ECEI: 9..12 (seq_data layout [bes | ecei | mc]).
BES_SLICE = slice(0, 9)
ECEI_SLICE = slice(9, 13)
COL = {"dt": 15, "carried_ti": 16, "stale_ti": 17, "carried_vt": 19, "stale_vt": 20}


def probe_quantities(x):
    """Per-step probe targets from one block's feature matrix x (L, 22)."""
    te_mean = x[:, ECEI_SLICE].mean(axis=1)
    bes = pd.DataFrame(x[:, BES_SLICE])
    activity = bes.rolling(10, min_periods=2).std(ddof=0).mean(axis=1).fillna(0.0).to_numpy()
    return {
        "te_mean": te_mean.astype(np.float64),
        "activity": activity.astype(np.float64),
        "staleness_ti": x[:, COL["stale_ti"]].astype(np.float64),
        "carried_ti": x[:, COL["carried_ti"]].astype(np.float64),
        "staleness_vt": x[:, COL["stale_vt"]].astype(np.float64),
        "carried_vt": x[:, COL["carried_vt"]].astype(np.float64),
    }


def collect(model, grid, dims, stats, names):
    """Latents + probe targets over every grid step of the named files."""
    zs_ti, zs_vt, quants, anchors = [], [], [], []
    with torch.no_grad():
        for name in names:
            if name not in grid:
                continue
            for blk in build_blocks(grid[name], dims, stats, per_shot_norm=PER_SHOT):
                x = torch.from_numpy(blk["x"]).unsqueeze(0)
                z_ti, z_vt = model.latents(x, None)
                zs_ti.append(z_ti[0].numpy())
                zs_vt.append(z_vt[0].numpy())
                quants.append(probe_quantities(blk["x"]))
                anchors.append(blk["x"][:, [COL["carried_ti"], COL["carried_vt"]]])
    z_ti = np.concatenate(zs_ti).astype(np.float64)
    z_vt = np.concatenate(zs_vt).astype(np.float64)
    q = {k: np.concatenate([d[k] for d in quants]) for k in quants[0]}
    anchor = np.concatenate(anchors).astype(np.float64)
    return z_ti, z_vt, q, anchor


def subsample(n, cap, rng):
    if n <= cap:
        return np.arange(n)
    return rng.choice(n, size=cap, replace=False)


def ols_r2(z_fit, y_fit, z_eval, y_eval):
    """Fit OLS (with intercept) on fit rows, return R^2 on eval rows."""
    A = np.column_stack([z_fit, np.ones(len(z_fit))])
    coef, *_ = np.linalg.lstsq(A, y_fit, rcond=None)
    pred = np.column_stack([z_eval, np.ones(len(z_eval))]) @ coef
    ss_res = float(((y_eval - pred) ** 2).sum())
    ss_tot = float(((y_eval - y_eval.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def probes_for(z_fit, z_eval, q_fit, q_eval, quantity_names):
    out = {}
    for qn in quantity_names:
        entry = {"full_latent_r2": round(ols_r2(z_fit, q_fit[qn], z_eval, q_eval[qn]), 4)}
        entry["per_dim_r2"] = [
            round(ols_r2(z_fit[:, k:k + 1], q_fit[qn], z_eval[:, k:k + 1], q_eval[qn]), 4)
            for k in range(z_fit.shape[1])
        ]
        out[qn] = entry
    return out


def one_run(variant, seed):
    out_dir = DATA / f".b3c_{variant}_s{seed}"
    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    stats = _load_stats(metrics)
    manifest = json.loads((DATA / f".b1_w2cut_split_s{seed}" / "split_manifest.json")
                          .read_text(encoding="utf-8"))

    model = SEQ_MODELS[variant]()
    model.load_state_dict(torch.load(out_dir / "weights" / "seq_lstm.pth", map_location="cpu"))
    model.eval()

    grid, dims = load_grid_files(DATA, DROP_STUCK, ti_spike_cut_ev=TI_SPIKE_CUT_EV)
    z_ti_tr, z_vt_tr, q_tr, _ = collect(model, grid, dims, stats, manifest["train_files"])
    z_ti_te, z_vt_te, q_te, anchor_te = collect(model, grid, dims, stats, manifest["test_files"])

    rng = np.random.default_rng(PROBE_SEED)
    fit_idx = subsample(len(z_ti_tr), MAX_FIT_STEPS, rng)
    eval_idx = subsample(len(z_ti_te), MAX_EVAL_STEPS, rng)
    sel = lambda arrs, idx: {k: v[idx] for k, v in arrs.items()}

    report = {
        "variant": variant, "seed": seed,
        "n_fit_steps": int(len(fit_idx)), "n_eval_steps": int(len(eval_idx)),
        "probes_z_ti": probes_for(z_ti_tr[fit_idx], z_ti_te[eval_idx],
                                  sel(q_tr, fit_idx), sel(q_te, eval_idx),
                                  ("te_mean", "activity", "staleness_ti", "carried_ti")),
        "probes_z_vt": probes_for(z_vt_tr[fit_idx], z_vt_te[eval_idx],
                                  sel(q_tr, fit_idx), sel(q_te, eval_idx),
                                  ("staleness_vt", "carried_vt")),
    }

    # (2) exact decomposition stats on TEST steps.
    w_ti = model.w_ti.weight.detach().numpy()[0].astype(np.float64)
    b_ti = float(model.w_ti.bias.detach())
    w_vt = model.w_vt.weight.detach().numpy()[0].astype(np.float64)
    b_vt = float(model.w_vt.bias.detach())
    decomp = {}
    for tag, z, w, b, anchor in (("ti", z_ti_te, w_ti, b_ti, anchor_te[:, 0]),
                                 ("vt", z_vt_te, w_vt, b_vt, anchor_te[:, 1])):
        contrib = z * w                       # (N, K) exact per-dim terms
        corr = contrib.sum(axis=1) + b
        total = anchor + corr
        decomp[tag] = {
            "readout_w": [round(float(v), 5) for v in w],
            "readout_bias": round(b, 5),
            "anchor_std": round(float(anchor.std()), 4),
            "correction_std": round(float(corr.std()), 4),
            "correction_share_of_pred_var": round(float(corr.var() / total.var()), 4)
            if total.var() > 0 else float("nan"),
            "mean_abs_contrib_per_dim": [round(float(np.abs(contrib[:, k]).mean()), 4)
                                         for k in range(contrib.shape[1])],
        }
    report["decomposition_test"] = decomp

    # (3) structural routing check on a real test block (CPU, bit-level).
    blk = None
    for name in manifest["test_files"]:
        if name in grid:
            cand = build_blocks(grid[name], dims, stats, per_shot_norm=PER_SHOT)
            if cand and cand[0]["x"].shape[0] >= 20:
                blk = cand[0]
                break
    x = torch.from_numpy(blk["x"]).unsqueeze(0)
    with torch.no_grad():
        base = model(x, None)
        x2 = x.clone()
        x2[..., :N_FAST_CHANNELS] += torch.randn_like(x2[..., :N_FAST_CHANNELS])
        pert = model(x2, None)
    report["routing_structural_check"] = {
        "vt_bit_identical_under_fast_perturbation": bool(torch.equal(base[..., 1], pert[..., 1])),
        "ti_responds_to_fast_channels": bool(not torch.equal(base[..., 0], pert[..., 0])),
    }

    (out_dir / "probe_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[b3p] s{seed}: probes + decomposition + routing check saved "
          f"({out_dir / 'probe_report.json'})", flush=True)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    args = ap.parse_args()

    reports = [one_run(args.variant, seed) for seed in args.seeds]
    ok_routing = all(r["routing_structural_check"]["vt_bit_identical_under_fast_perturbation"]
                     and r["routing_structural_check"]["ti_responds_to_fast_channels"]
                     for r in reports)
    summary = {
        "variant": args.variant,
        "routing_structural_check_all_pass": ok_routing,
        "per_seed": reports,
    }
    out_path = DATA / f".b3_probe_summary_{args.variant}.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[b3p] summary saved {out_path} (routing all-pass: {ok_routing})", flush=True)


if __name__ == "__main__":
    main()
