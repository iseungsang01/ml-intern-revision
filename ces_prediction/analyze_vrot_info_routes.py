"""Which information routes are open to a V_rot imputer? (THESIS_RESULTS.md §8au)

Descriptive only: no model, no split, no TEST. It asks three questions that decide which
multimodal model family (PREREGISTRATION_B12.md) could even in principle help CES_VT:

1. Cross-target route -- when CES_VT is missing, is CES_TI observed at the same instant,
   and does CES_TI carry information about CES_VT (linear, rank, and binned eta^2)?
   A joint generative imputer (diffusion over [TI, VT]) can only exploit this route.
2. Shot-level route -- how much of CES_VT variance is between-shot (a per-shot offset a
   shot encoder / neural-process latent could carry) vs within-shot?
3. Label-scarcity route -- how many rows have inputs but no CES_VT label (what masked
   pretraining / semi-supervision can use), and how many shots have almost no CES_VT
   context (where a shot prior, not same-shot interpolation, would have to work)?

Data treatment is the confirmed protocol's (held-free, both populations: TI spike cut
3 keV and inclusive). Output: data/.vrot_info_routes.json.
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "experiments" / "seq")]
from seq_data import load_grid_files  # noqa: E402

DATA_DIR = Path(__import__("os").getenv("CES_DATA_DIR", ROOT.parent / "data"))
OUT = ROOT.parent / "data" / ".vrot_info_routes.json"
MIN_ROWS = 5


def eta_squared(x, y, n_bins=20):
    """Correlation ratio of y on quantile bins of x: share of var(y) any function of x explains."""
    edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    idx = np.clip(np.digitize(x, edges[1:-1]), 0, n_bins - 1)
    grand = y.mean()
    between = sum((idx == i).sum() * (y[idx == i].mean() - grand) ** 2
                  for i in range(n_bins) if (idx == i).any())
    return float(between / ((y - grand) ** 2).sum())


def routes(grid):
    n = ti_obs = vt_obs = ti_only = 0
    t_all, v_all, t_w, v_w, t_b, v_b = [], [], [], [], [], []
    vt_series, vt_counts = [], []
    for arr in grid.values():
        ti = ~np.isnan(arr[:, 1])
        vt = ~np.isnan(arr[:, 2])
        n += len(arr)
        ti_obs += int(ti.sum())
        vt_obs += int(vt.sum())
        ti_only += int((ti & ~vt).sum())
        vt_counts.append(int(vt.sum()))
        if vt.sum() > MIN_ROWS:
            vt_series.append(arr[vt, 2].astype(np.float64))
        both = ti & vt
        if both.sum() > MIN_ROWS:
            t, v = arr[both, 1].astype(np.float64), arr[both, 2].astype(np.float64)
            t_all.append(t); v_all.append(v)
            t_w.append(t - t.mean()); v_w.append(v - v.mean())
            t_b.append(t.mean()); v_b.append(v.mean())
    t, v = np.concatenate(t_all), np.concatenate(v_all)
    tw, vw = np.concatenate(t_w), np.concatenate(v_w)
    pooled = np.concatenate(vt_series)
    within = np.concatenate([s - s.mean() for s in vt_series])
    counts = np.asarray(vt_counts)
    return {
        "rows": n, "ti_observed": ti_obs, "vt_observed": vt_obs,
        "cross_target": {
            "vt_gaps_with_ti_observed_frac": ti_only / (n - vt_obs),
            "pearson_pooled": float(np.corrcoef(t, v)[0, 1]),
            "spearman_pooled": float(spearmanr(t, v)[0]),
            "pearson_within_shot": float(np.corrcoef(tw, vw)[0, 1]),
            "pearson_between_shot_means": float(np.corrcoef(t_b, v_b)[0, 1]),
            "eta2_vt_given_ti_20bins": eta_squared(t, v),
            "eta2_vt_given_ti_within_shot_20bins": eta_squared(tw, vw),
            "n_rows_both": int(len(t)), "n_shots_both": len(t_b),
        },
        "shot_level": {
            "vt_within_shot_variance_share": float(within.var() / pooled.var()),
            "vt_between_shot_variance_share": float(1 - within.var() / pooled.var()),
            "n_shots": len(vt_series),
        },
        "label_scarcity": {
            "rows_without_vt_label_frac": 1 - vt_obs / n,
            "rows_without_ti_label_frac": 1 - ti_obs / n,
            "shots_total": int(len(counts)),
            "shots_vt_labels_0": int((counts == 0).sum()),
            "shots_vt_labels_1_to_20": int(((counts > 0) & (counts <= 20)).sum()),
            "shots_vt_labels_over_20": int((counts > 20).sum()),
        },
    }


def main():
    result = {}
    for name, cut in (("cut_3keV", 3000.0), ("inclusive", 0.0)):
        grid, _ = load_grid_files(DATA_DIR, drop_stuck_targets=True, ti_spike_cut_ev=cut)
        result[name] = routes(grid)
        ct = result[name]["cross_target"]
        sl = result[name]["shot_level"]
        ls = result[name]["label_scarcity"]
        print(f"[{name}] VT gaps with TI observed {ct['vt_gaps_with_ti_observed_frac']:.3f} | "
              f"r={ct['pearson_pooled']:.3f} rho={ct['spearman_pooled']:.3f} "
              f"eta2={ct['eta2_vt_given_ti_20bins']:.4f} (within {ct['eta2_vt_given_ti_within_shot_20bins']:.4f}) | "
              f"VT between-shot share {sl['vt_between_shot_variance_share']:.3f} | "
              f"rows w/o VT {ls['rows_without_vt_label_frac']:.3f} | "
              f"shots VT=0/1-20/>20: {ls['shots_vt_labels_0']}/{ls['shots_vt_labels_1_to_20']}/{ls['shots_vt_labels_over_20']}")
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
