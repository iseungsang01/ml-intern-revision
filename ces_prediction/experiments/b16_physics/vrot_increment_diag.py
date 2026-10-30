"""Why do two level priors stall at 1.18x persistence in the V_rot median? (sec. 8bk, named next)

TRAIN shots only (the four frozen split manifests), no model, no TEST row. Data treatment is the
confirmed protocol's: held repeats NaN'd (held-free), T_i spikes > CES_TI_SPIKE_CUT_EV cut first;
increments only across consecutive 10 ms grid steps inside one contiguous block.

Three readouts per target and population:
  1. Increment lag-1 correlation rho1 = corr(d_t, d_{t+1}): Pearson (tail-weighted), Spearman, and
     the bulk sign concordance P[sign d_{t+1} = sign d_t | both |d| <= median |d|].
     White noise on a random walk gives rho1 < 0 (pure noise -0.5); a continuing drift gives > 0.
  2. Simple causal predictors on rows whose previous two rows are observed, median AE / persistence:
     exponential smoothing x <- x + k (y - x) (k = 0.3, 0.5, 0.7) and persistence + beta * last slope
     (beta = 0.25, 0.5). Below 1 = that structure beats carrying the last value on the median row.
  3. Scale: robust increment scatter (1.4826 * MAD(d) / sqrt 2) against the anchors' learned sigma_n.

Reading (fixed before the first run):
  * slope predictor < 0.99 and smoothing >= 1  -> the median rows trend: next cell = level + velocity.
  * smoothing < 0.99 and slope >= 1            -> the bulk is noise around a level, and the anchors'
    sigma_n is too large for it: next = heavy-tailed (Student-t) measurement noise in the anchor.
  * both >= 0.99                                -> persistence is median-optimal at 10 ms in the bulk
    (bulk increments are a random walk); the P02 0.921x bound is unreachable by any causal filter
    of this history, and the 1.18x is pure shrinkage cost.
Output: data/.b19_vrot_diag.json.  Usage (repo root): py ces_prediction/experiments/b16_physics/vrot_increment_diag.py
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "ces_prediction" / "experiments"))
sys.path.insert(0, str(ROOT / "ces_prediction" / "experiments" / "seq"))
sys.path.insert(0, str(ROOT / "ces_prediction"))
from runner_common import SEEDS, SPLIT_SRC  # noqa: E402
from seq_data import load_grid_files  # noqa: E402

DATA = ROOT / "data"
STEP = 0.01
K_SMOOTH = (0.3, 0.5, 0.7)
BETA_SLOPE = (0.25, 0.5)
LEARNED_SIGMA_N = {"CES_TI": {"ou_or_jump_cut": "118-124 eV", "jump_incl": "191-201 eV"},
                   "CES_VT": {"jump": "25-33 km/s"}}


def train_names():
    names = set()
    for s in SEEDS:
        m = json.loads((SPLIT_SRC[s] / "split_manifest.json").read_text(encoding="utf-8"))
        names.update(m["train_files"])
    return names


def triples(t, y):
    """(y[i-2], y[i-1], y[i]) for i with three consecutive observed 10 ms steps."""
    ok = np.isfinite(y)
    step = np.isclose(np.diff(t), STEP, atol=1e-3)
    i = np.arange(2, len(y))
    keep = ok[i] & ok[i - 1] & ok[i - 2] & step[i - 1] & step[i - 2]
    i = i[keep]
    return y[i - 2], y[i - 1], y[i]


def smooth_pred(t, y, k):
    """Causal exponential smoothing over observed values; prediction for row i uses rows < i.
    Restarted at every gap (non-10 ms step or missing value) from the next observation."""
    pred = np.full_like(y, np.nan)
    x = np.nan
    for i in range(len(y)):
        if i > 0 and not (np.isfinite(y[i - 1]) and abs(t[i] - t[i - 1] - STEP) < 1e-3):
            x = np.nan if not np.isfinite(y[i - 1]) else y[i - 1]
        pred[i] = x
        if np.isfinite(y[i]):
            x = y[i] if not np.isfinite(x) else x + k * (y[i] - x)
    return pred


def analyse(grid, col):
    d1, d2, err = [], [], {"persistence": [], **{f"smooth_k{k}": [] for k in K_SMOOTH},
                            **{f"slope_b{b}": [] for b in BETA_SLOPE}}
    for arr in grid.values():
        t, y = arr[:, 0].astype(np.float64), arr[:, col].astype(np.float64)
        a, b, c = triples(t, y)
        if len(c) == 0:
            continue
        d1.append(b - a)
        d2.append(c - b)
        err["persistence"].append(np.abs(c - b))
        for beta in BETA_SLOPE:
            err[f"slope_b{beta}"].append(np.abs(c - (b + beta * (b - a))))
        ok = np.isfinite(y)
        step = np.isclose(np.diff(t), STEP, atol=1e-3)
        i = np.arange(2, len(y))
        rows = i[ok[i] & ok[i - 1] & ok[i - 2] & step[i - 1] & step[i - 2]]
        for k in K_SMOOTH:
            p = smooth_pred(t, y, k)
            err[f"smooth_k{k}"].append(np.abs(y[rows] - p[rows]))
    d1, d2 = np.concatenate(d1), np.concatenate(d2)
    med = np.median(np.abs(np.concatenate([d1, d2])))
    bulk = (np.abs(d1) <= med) & (np.abs(d2) <= med) & (d1 != 0) & (d2 != 0)
    e = {k: np.concatenate(v) for k, v in err.items()}
    base = float(np.median(e["persistence"]))
    mad = float(np.median(np.abs(d1 - np.median(d1))))
    return {
        "n_triples": int(len(d1)),
        "rho1_pearson": float(np.corrcoef(d1, d2)[0, 1]),
        "rho1_spearman": float(spearmanr(d1, d2).statistic),
        "bulk_sign_concordance": float(np.mean(np.sign(d1[bulk]) == np.sign(d2[bulk]))),
        "n_bulk_pairs": int(bulk.sum()),
        "median_abs_increment": float(med),
        "robust_noise_scale": 1.4826 * mad / np.sqrt(2.0),
        "rms_noise_scale_rice": float(np.sqrt(np.mean(d1 ** 2) / 2.0)),
        "median_ae_persistence": base,
        "median_ae_over_persistence": {k: float(np.median(v)) / base for k, v in e.items() if k != "persistence"},
    }


def reading(r):
    sm = min(v for k, v in r["median_ae_over_persistence"].items() if k.startswith("smooth"))
    sl = min(v for k, v in r["median_ae_over_persistence"].items() if k.startswith("slope"))
    if sl < 0.99 and sm >= 1.0:
        return "median rows trend -> level+velocity cell"
    if sm < 0.99 and sl >= 1.0:
        return "bulk is noise around a level, sigma_n too large -> heavy-tailed measurement noise in the anchor"
    if sm >= 0.99 and sl >= 0.99:
        return "persistence is median-optimal in the bulk -> 1.18x is shrinkage cost; P02 0.921x unreachable"
    return "mixed (both structures beat persistence)"


def main():
    names = train_names()
    out = {"train_files": len(names), "learned_sigma_n": LEARNED_SIGMA_N}
    for pop, cut in (("cut", 3000.0), ("incl", 0.0)):
        grid, _ = load_grid_files(DATA, drop_stuck_targets=True, ti_spike_cut_ev=cut)
        grid = {k: v for k, v in grid.items() if k in names}
        out[pop] = {}
        for col, tgt in ((1, "CES_TI"), (2, "CES_VT")):
            r = analyse(grid, col)
            r["reading"] = reading(r)
            out[pop][tgt] = r
            print(f"[diag] {pop} {tgt}: n={r['n_triples']} rho1 P {r['rho1_pearson']:+.3f} S {r['rho1_spearman']:+.3f}"
                  f" bulk-sign {r['bulk_sign_concordance']:.3f} | robust sigma {r['robust_noise_scale']:.2f}"
                  f" rice {r['rms_noise_scale_rice']:.2f} | " + " ".join(f"{k} {v:.3f}" for k, v in r["median_ae_over_persistence"].items())
                  + f" | {r['reading']}", flush=True)
    (DATA / ".b19_vrot_diag.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {DATA / '.b19_vrot_diag.json'}")


if __name__ == "__main__":
    if not any(DATA.glob("s*.csv")):
        raise SystemExit("FATAL: no shot CSVs in data/ -- real data required.")
    os.environ.setdefault("CES_TI_SPIKE_CUT_EV", "0")
    main()
