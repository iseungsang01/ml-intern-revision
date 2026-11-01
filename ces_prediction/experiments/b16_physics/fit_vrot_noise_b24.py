"""B.24 step 1 (PREREGISTRATION_B24.md sec. 2): fit the V_rot measurement model (sigma_n, nu) on TRAIN.

Same objective and recursion as B.23 (fit_vrot_jump_b23.scan_loglik), with (lam, sigma_J) held at
the B.23 TRAIN fit of each split and (r = sigma_n^2, nu) as the grid dimensions.
Output: data/.b24_vrot_noise_fit.json.  Usage (repo root):  py ces_prediction/experiments/b16_physics/fit_vrot_noise_b24.py
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fit_vrot_jump_b23 import (DATA, MANIFEST, SEEDS, build_blocks, fit_stats, load_grid_files,  # noqa: E402
                               scan_loglik, vt_series)

SN_GRID = (0.03, 0.04, 0.05, 0.06, 0.07, 0.085, 0.10, 0.13, 0.17)   # target sd
NU_GRID = (2.2, 2.5, 3.0, 4.0, 6.0, 10.0, 30.0)
BASE = {"sigma_n": 0.060, "nu": 3.0}


def main():
    b23 = json.loads((DATA / ".b23_vrot_jump_fit.json").read_text(encoding="utf-8"))["fits"]
    grid, dims = load_grid_files(DATA, True, 0.0)   # the T_i cut never touches V_rot
    sn = np.array([a for a in SN_GRID for _ in NU_GRID])
    nu = np.array([b for _ in SN_GRID for b in NU_GRID])
    out = {"grid": {"sigma_n": list(SN_GRID), "nu": list(NU_GRID)}, "base": BASE, "fits": {}}
    for pop, tmpl in MANIFEST.items():
        for s in SEEDS:
            names = [n for n in json.loads((DATA / tmpl.format(s=s) / "split_manifest.json")
                                           .read_text(encoding="utf-8"))["train_files"] if n in grid]
            stats = fit_stats(grid, dims, names)
            blocks = [b for n in names for b in build_blocks(grid[n], dims, stats, per_shot_norm=True)]
            carried, obs_prev = vt_series(blocks)
            f23 = b23[f"{pop}_s{s}"]
            G = len(sn)
            ll, n = scan_loglik(carried, obs_prev, np.full(G, f23["lam_per_step"]), np.full(G, f23["sigma_j"] ** 2),
                                r=(sn ** 2)[:, None], nu=nu[:, None])
            k = int(np.argmax(ll))
            base = int(np.flatnonzero((sn == BASE["sigma_n"]) & (nu == BASE["nu"]))[0])
            std = float(stats["target"]["std"][1])
            fit = {"sigma_n": float(sn[k]), "nu": float(nu[k]), "sigma_n_kms": float(sn[k]) * std,
                   "lam_per_step": f23["lam_per_step"], "sigma_j": f23["sigma_j"],
                   "loglik_per_obs": float(ll[k] / n), "base_loglik_per_obs": float(ll[base] / n), "n_obs": n,
                   "at_grid_edge": bool(sn[k] in (SN_GRID[0], SN_GRID[-1]) or nu[k] in (NU_GRID[0], NU_GRID[-1])),
                   "surface": [[float(ll[i * len(NU_GRID) + q] / n) for q in range(len(NU_GRID))]
                               for i in range(len(SN_GRID))]}
            if abs(fit["base_loglik_per_obs"] - f23["loglik_per_obs"]) > 1e-9:
                raise SystemExit(f"FATAL: base point does not reproduce the B.23 optimum ({pop} s{s})")
            out["fits"][f"{pop}_s{s}"] = fit
            print(f"[b24-fit] {pop} s{s}: sigma_n {fit['sigma_n']:.3f} sd ({fit['sigma_n_kms']:.2f} km/s), nu {fit['nu']:.1f}, "
                  f"ll/obs {fit['loglik_per_obs']:+.4f} (B.23 base {fit['base_loglik_per_obs']:+.4f}), edge={fit['at_grid_edge']}",
                  flush=True)
    (DATA / ".b24_vrot_noise_fit.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {DATA / '.b24_vrot_noise_fit.json'}")


if __name__ == "__main__":
    main()
