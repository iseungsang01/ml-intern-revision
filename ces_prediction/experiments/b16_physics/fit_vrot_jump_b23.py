"""B.23 step 1 (PREREGISTRATION_B23.md sec. 2): fit the V_rot jump process of the B.19 anchor on TRAIN.

The B.19 hold-or-jump cell with Student-t measurement noise is scored here by its exact one-step
predictive log-likelihood on each split's TRAIN shots -- no model, no VAL or TEST row. At every
observation the predictive density of the new value is the two-hypothesis mixture the filter
itself uses: (1 - h) t_nu(e; S0) + h t_nu(e; S1), S0 = P + r, S1 = P + sigma_J^2 + r, h = jump
hazard since the last update. The recursion is a float64 numpy copy of model_seq_v4k._jump_scan.

Fitted: (lam, sigma_J) on a fixed grid. Held at the registered B.19 values: sigma_n = 0.060 target
sd (BULK_SIGMA_N), nu = 3, sigma_d = 0.05, p0 = 1. The first observation of every block is not
scored (its prior is the diffuse p0, not the jump process).
Input rows are seq_data.build_blocks of the confirmed protocol (held-free, per-shot norm of the
fast channels -- irrelevant here; targets on the TRAIN-file global stats).
Output: data/.b23_vrot_jump_fit.json.  Usage (repo root):  py ces_prediction/experiments/b16_physics/fit_vrot_jump_b23.py
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.special import gammaln

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "ces_prediction" / "experiments" / "seq"))
from seq_data import N_FAST_CHANNELS, build_blocks, fit_stats, load_grid_files  # noqa: E402

DATA = ROOT / "data"
SEEDS = (42, 1, 7, 123)
MANIFEST = {"cut": ".b1_w2cut_split_s{s}", "incl": ".b1_manifest_s{s}"}
LAM_GRID = (0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2, 0.3)          # per 10 ms step
SJ_GRID = (0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0)                                    # target sd
REGISTERED = {"lam": 0.05, "sigma_j": 1.0}
SIGMA_N, NU, SIGMA_D, P0 = 0.060, 3.0, 0.05, 1.0
COL_VT_CARRIED = N_FAST_CHANNELS + 1 + 3   # seq_data layout [fast | dt | TI(c,s,h) | VT(c,s,h)]
GRID_DT_S = 0.01


def log_t(e2, S, nu):
    """log Student-t density of an innovation with squared value e2, scale^2 S, dof nu."""
    return (gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(nu * math.pi * S)
            - 0.5 * (nu + 1) * np.log1p(e2 / (nu * S)))


def scan_loglik(carried, obs_prev, lam, j, r=SIGMA_N ** 2, nu=NU, d=SIGMA_D ** 2, p0=P0, anchors=None):
    """Summed predictive log-lik and the number of scored observations.

    carried, obs_prev: (B, L); lam, j: (G,) hypotheses evaluated in one vectorised pass.
    Arithmetic of the filter state is model_seq_v4k._jump_scan with student=True."""
    G, (B, L) = len(lam), carried.shape
    lam, j = lam[:, None], j[:, None]
    x = np.zeros((G, B)); P = np.full((G, B), p0); h = np.zeros((G, B))
    seen = np.zeros(B, bool)
    total, n = np.zeros(G), 0
    for t in range(L):
        if t > 0:
            upd = obs_prev[:, t] > 0.5
            if upd.any():
                e = carried[None, :, t] - x
                e2 = e * e
                S0, S1 = P + r, P + j + r
                hh = np.clip(h, 1e-6, 1 - 1e-6)
                l0, l1 = log_t(e2, S0, nu), log_t(e2, S1, nu)
                mix = np.logaddexp(np.log1p(-hh) + l0, np.log(hh) + l1)
                score = upd & seen
                total += (mix * score[None]).sum(axis=1)
                n += int(score.sum())
                ll = l1 - l0  # = the model's log-likelihood ratio (gamma terms cancel)
                w1 = 1.0 / (1.0 + np.exp(-(np.log(hh) - np.log1p(-hh) + ll)))
                R0 = r * (nu + e2 / S0) / (nu + 1)
                R1 = r * (nu + e2 / S1) / (nu + 1)
                T0, T1 = P + R0, P + j + R1
                x0, x1 = x + P / T0 * e, x + (P + j) / T1 * e
                xm = (1 - w1) * x0 + w1 * x1
                Pm = (1 - w1) * (P * R0 / T0 + (x0 - xm) ** 2) + w1 * ((P + j) * R1 / T1 + (x1 - xm) ** 2)
                x = np.where(upd[None], xm, x)
                P = np.where(upd[None], Pm, P)
                h = np.where(upd[None], 0.0, h)
                seen |= upd
        P = P + d
        h = 1 - (1 - h) * (1 - lam)
        if anchors is not None:
            anchors.append(x.copy())
    return total, n


def vt_series(blocks):
    L = max(len(b["y"]) for b in blocks)
    carried = np.zeros((len(blocks), L)); obs_prev = np.zeros((len(blocks), L))
    for i, b in enumerate(blocks):
        n = len(b["y"])
        carried[i, :n] = b["x"][:, COL_VT_CARRIED]
        obs_prev[i, 1:n] = b["mask"][:-1, 1]
    return carried, obs_prev


def main():
    grid, dims = load_grid_files(DATA, True, 0.0)   # the T_i cut never touches V_rot
    lam = np.array([l for l in LAM_GRID for _ in SJ_GRID])
    sj = np.array([s for _ in LAM_GRID for s in SJ_GRID])
    out = {"grid": {"lam_per_step": list(LAM_GRID), "sigma_j": list(SJ_GRID)}, "fixed": {
        "sigma_n": SIGMA_N, "nu": NU, "sigma_d": SIGMA_D, "p0": P0}, "registered": REGISTERED, "fits": {}}
    for pop, tmpl in MANIFEST.items():
        for s in SEEDS:
            names = [n for n in json.loads((DATA / tmpl.format(s=s) / "split_manifest.json")
                                           .read_text(encoding="utf-8"))["train_files"] if n in grid]
            stats = fit_stats(grid, dims, names)
            blocks = [b for n in names for b in build_blocks(grid[n], dims, stats, per_shot_norm=True)]
            carried, obs_prev = vt_series(blocks)
            ll, n = scan_loglik(carried, obs_prev, lam, sj ** 2)
            k = int(np.argmax(ll))
            reg = int(np.flatnonzero((lam == REGISTERED["lam"]) & (sj == REGISTERED["sigma_j"]))[0])
            std = float(stats["target"]["std"][1])
            fit = {"lam_per_step": float(lam[k]), "sigma_j": float(sj[k]),
                   "jump_rate_hz": -math.log(1 - lam[k]) / GRID_DT_S, "sigma_j_kms": float(sj[k]) * std,
                   "loglik_per_obs": float(ll[k] / n), "registered_loglik_per_obs": float(ll[reg] / n),
                   "n_obs": n, "n_blocks": len(blocks), "target_std_kms": std,
                   "at_grid_edge": bool(lam[k] in (LAM_GRID[0], LAM_GRID[-1]) or sj[k] in (SJ_GRID[0], SJ_GRID[-1])),
                   "surface": [[float(ll[i * len(SJ_GRID) + q] / n) for q in range(len(SJ_GRID))]
                               for i in range(len(LAM_GRID))]}
            out["fits"][f"{pop}_s{s}"] = fit
            print(f"[b23-fit] {pop} s{s}: lam {fit['lam_per_step']:.3f}/step ({fit['jump_rate_hz']:.2f} Hz), "
                  f"sigma_J {fit['sigma_j']:.2f} sd ({fit['sigma_j_kms']:.1f} km/s), ll/obs {fit['loglik_per_obs']:+.4f} "
                  f"(registered {fit['registered_loglik_per_obs']:+.4f}), n={n}, edge={fit['at_grid_edge']}", flush=True)
    (DATA / ".b23_vrot_jump_fit.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {DATA / '.b23_vrot_jump_fit.json'}")


if __name__ == "__main__":
    main()
