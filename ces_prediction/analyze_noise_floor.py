"""Measure the CES measurement-noise floor, so the nowcaster's RMSE can be read
against an absolute reference instead of only against other estimators.

Why this exists
---------------
Every ceiling statement in THESIS_RESULTS.md so far is *relative*: a 21k-parameter
rung matches the backbone (S8z), a 26x width sweep is flat (S8aa), three operator
families tie (S8ag).  Those say "a bigger estimator does not help"; none of them
says how much of the residual is even reducible.  The target is a photon-integrated
spectral fit, so each scored value carries its own measurement noise, and a perfect
predictor of the true physical T_i would still score

    RMSE^2 = sigma_meas^2 + (physics error)^2

against it.  This script estimates sigma_meas from the observed series itself, with
difference-based estimators that are unbiased for white noise on a smooth signal and
become progressively insensitive to that signal as the difference order rises:

    order 1 (Rice)        sigma^2 = mean(d1^2) /  2      d1 = x[i+1] - x[i]
    order 2 (GSJS)        sigma^2 = mean(d2^2) /  6      d2 = x[i-1] - 2x[i] + x[i+1]
    order 3               sigma^2 = mean(d3^2) / 20      binomial (1,-3,3,-1)
    order 4               sigma^2 = mean(d4^2) / 70      binomial (1,-4,6,-4,1)

Each is an UPPER bound on the measurement noise: whatever real physics moves faster
than the difference operator can annihilate is counted as noise.  The sequence is
non-increasing in the signal-bias term, so its convergence (or lack of it) is the
result.  A robust MAD variant of order 1 and a semivariogram nugget are reported
alongside, because the CES population carries single-sample fit-failure outliers
that inflate any mean-of-squares estimator.

Data treatment follows the confirmed protocol exactly (dataset.py's rules):
spikes are cut BEFORE held detection, held (forward-filled) repeats are removed,
and only consecutive same-block grid steps enter a difference.

Run from the repo root:  py ces_prediction/analyze_noise_floor.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

STUCK_GAP_SECONDS = 0.5
TARGETS = ("CES_TI", "CES_VT")
UNITS = {"CES_TI": "eV", "CES_VT": "km/s"}

# Binomial difference filters: coefficients and their sum of squares.
DIFF_FILTERS = {
    1: (np.array([1.0, -1.0]), 2.0),
    2: (np.array([1.0, -2.0, 1.0]), 6.0),
    3: (np.array([1.0, -3.0, 3.0, -1.0]), 20.0),
    4: (np.array([1.0, -4.0, 6.0, -4.0, 1.0]), 70.0),
}

MAX_VARIOGRAM_LAG = 12


def data_dir() -> Path:
    return Path(os.environ.get("CES_DATA_DIR", "data"))


def _stuck_repeat_mask(values: np.ndarray, block: np.ndarray) -> np.ndarray:
    """Held rule, byte-for-byte the same logic as dataset._stuck_repeat_mask."""
    stuck = np.zeros(values.shape, dtype=bool)
    last_val = np.nan
    last_block = -1
    for i, v in enumerate(values):
        if np.isnan(v):
            continue
        if block[i] == last_block and not np.isnan(last_val) and v == last_val:
            stuck[i] = True
        else:
            last_val = v
            last_block = block[i]
    return stuck


def clean_target(df: pd.DataFrame, col: str, cut_ev: float) -> np.ndarray:
    """Return the target column with spikes and held repeats set to NaN."""
    values = df[col].to_numpy(dtype=float).copy()
    if col == "CES_TI" and cut_ev > 0:
        values[values > cut_ev] = np.nan
    time = df["time"].to_numpy(dtype=float)
    block = np.concatenate(([0], np.cumsum(np.diff(time) >= STUCK_GAP_SECONDS)))
    values[_stuck_repeat_mask(values, block)] = np.nan
    return values


def consecutive_runs(values: np.ndarray, time: np.ndarray, step: float):
    """Yield arrays of observed values that sit on consecutive grid steps.

    A difference estimator is only meaningful on an evenly-spaced stretch, so a
    run breaks whenever a value is missing or the time step is not one grid unit.
    """
    run: list[float] = []
    for i, v in enumerate(values):
        if np.isnan(v):
            if len(run) > 1:
                yield np.asarray(run)
            run = []
            continue
        if run and not (0.5 * step < time[i] - time[i - 1] < 1.5 * step):
            if len(run) > 1:
                yield np.asarray(run)
            run = []
        run.append(v)
    if len(run) > 1:
        yield np.asarray(run)


def collect(cut_ev: float) -> dict:
    files = sorted(data_dir().glob("*.csv"))
    if not files:
        raise SystemExit(
            "no shot CSVs found -- point CES_DATA_DIR at the real data folder"
        )

    acc = {
        t: {
            "sq": {k: 0.0 for k in DIFF_FILTERS},
            "n": {k: 0 for k in DIFF_FILTERS},
            "abs_d1": [],
            "abs_d4": [],
            "vario_sq": {h: 0.0 for h in range(1, MAX_VARIOGRAM_LAG + 1)},
            "vario_n": {h: 0 for h in range(1, MAX_VARIOGRAM_LAG + 1)},
            "n_runs": 0,
            "n_obs": 0,
            "run_len_max": 0,
            "values": [],
        }
        for t in TARGETS
    }
    step = None

    for path in files:
        df = pd.read_csv(path, usecols=["time", *TARGETS])
        if df.empty:
            continue
        time = df["time"].to_numpy(dtype=float)
        if step is None and len(time) > 2:
            step = float(np.median(np.diff(time)))
        for target in TARGETS:
            values = clean_target(df, target, cut_ev)
            a = acc[target]
            for run in consecutive_runs(values, time, step):
                a["n_runs"] += 1
                a["n_obs"] += run.size
                a["run_len_max"] = max(a["run_len_max"], int(run.size))
                a["values"].append(run)
                for order, (coef, norm) in DIFF_FILTERS.items():
                    if run.size > coef.size - 1:
                        d = np.convolve(run, coef[::-1], mode="valid")
                        a["sq"][order] += float(np.sum(d * d) / norm)
                        a["n"][order] += d.size
                        if order == max(DIFF_FILTERS):
                            a["abs_d4"].append(np.abs(d))
                if run.size > 1:
                    a["abs_d1"].append(np.abs(np.diff(run)))
                for h in range(1, MAX_VARIOGRAM_LAG + 1):
                    if run.size > h:
                        d = run[h:] - run[:-h]
                        a["vario_sq"][h] += float(np.sum(d * d))
                        a["vario_n"][h] += d.size

    return {"acc": acc, "step": step, "n_files": len(files)}


def nugget_from_variogram(gamma: dict[int, float], step: float) -> dict:
    """Fit gamma(h) = c0 + a * h**b on the short lags and read c0 at h -> 0."""
    lags = np.array(sorted(gamma), dtype=float)
    vals = np.array([gamma[int(h)] for h in lags], dtype=float)
    use = lags <= 10
    lags, vals = lags[use], vals[use]

    best = None
    for b in np.linspace(0.2, 2.0, 91):
        x = lags**b
        design = np.vstack([np.ones_like(x), x]).T
        coef, *_ = np.linalg.lstsq(design, vals, rcond=None)
        resid = float(np.sum((design @ coef - vals) ** 2))
        if coef[0] >= 0 and (best is None or resid < best[0]):
            best = (resid, float(coef[0]), float(coef[1]), float(b))
    if best is None:
        return {"nugget_var": float("nan"), "exponent": float("nan")}
    _, c0, a, b = best
    return {
        "nugget_var": c0,
        "slope": a,
        "exponent": b,
        "sigma": float(np.sqrt(max(c0, 0.0))),
        "gamma_at_one_step_ms": float(step * 1000.0),
    }


def summarize(bundle: dict, cut_ev: float) -> dict:
    step = bundle["step"]
    out = {
        "n_files": bundle["n_files"],
        "grid_step_ms": round(step * 1000.0, 4),
        "cut_ev": cut_ev,
        "targets": {},
    }
    for target, a in bundle["acc"].items():
        orders = {}
        for order in sorted(DIFF_FILTERS):
            n = a["n"][order]
            var = a["sq"][order] / n if n else float("nan")
            orders[order] = {
                "sigma": float(np.sqrt(var)) if n else float("nan"),
                "n_differences": int(n),
            }
        abs_d1 = np.concatenate(a["abs_d1"]) if a["abs_d1"] else np.array([])
        mad_sigma = (
            float(1.4826 * np.median(abs_d1) / np.sqrt(2.0)) if abs_d1.size else float("nan")
        )
        gamma = {
            h: 0.5 * a["vario_sq"][h] / a["vario_n"][h]
            for h in a["vario_sq"]
            if a["vario_n"][h]
        }
        values = np.concatenate(a["values"]) if a["values"] else np.array([])

        # Separate the bulk of the noise from the outlier tail.  The same 1% of
        # rows that S8ab found carrying 70-83% of every arm's squared error also
        # dominates any mean-of-squares noise estimate, so report the trimmed
        # version and the mass the tail carries.
        abs_d4 = np.concatenate(a["abs_d4"]) if a["abs_d4"] else np.array([])
        trimmed = {}
        tail_mass = {}
        if abs_d4.size:
            sq4 = abs_d4**2
            total = float(sq4.sum())
            order4_norm = DIFF_FILTERS[max(DIFF_FILTERS)][1]
            for pct in (99.0, 95.0):
                keep = abs_d4 <= np.percentile(abs_d4, pct)
                trimmed[str(pct)] = float(
                    np.sqrt(sq4[keep].mean() / order4_norm)
                )
            for pct, label in ((99.0, "top1pct"), (95.0, "top5pct")):
                cut = abs_d4 > np.percentile(abs_d4, pct)
                tail_mass[label] = float(sq4[cut].sum() / total) if total else float("nan")

        out["targets"][target] = {
            "unit": UNITS[target],
            "n_runs": a["n_runs"],
            "n_observed_in_runs": a["n_obs"],
            "longest_run": a["run_len_max"],
            "value_std": float(values.std()) if values.size else float("nan"),
            "value_p50": float(np.median(values)) if values.size else float("nan"),
            "difference_estimators": orders,
            "robust_order1_mad_sigma": mad_sigma,
            "trimmed_order4_sigma": trimmed,
            "order4_squared_mass_in_tail": tail_mass,
            "semivariogram": {str(h): gamma[h] for h in sorted(gamma)},
            "variogram_nugget": nugget_from_variogram(gamma, step),
        }
    return out


# Confirmed-protocol RMSE of the arms, physical units, TEST seed 42, genuine
# measurements only.  Source: docs/paper/main_ko.tex table tab:ladder (which is
# generated from the frozen artifacts).  Kept here only to print the ratio.
LADDER_RMSE = {
    "cut": {
        "CES_TI": {"seq_v2": 157.8, "gp_causal": 164.3, "pchip": 173.6, "persistence": 197.2},
        "CES_VT": {"seq_v2": 23.6, "gp_causal": 28.8, "pchip": 30.2, "persistence": 33.4},
    },
    "inclusive": {
        "CES_TI": {"seq_v2": 363.0, "gp_causal": 394.6, "pchip": 412.4, "persistence": 478.0},
        "CES_VT": {"seq_v2": 23.7, "gp_causal": 28.8, "pchip": 30.2, "persistence": 33.4},
    },
}


def report(summary: dict, population: str) -> None:
    print("")
    print("=" * 78)
    print("CES measurement-noise floor -- population: %s (cut_ev=%s)"
          % (population, summary["cut_ev"]))
    print("files=%d  grid step=%.2f ms" % (summary["n_files"], summary["grid_step_ms"]))
    print("=" * 78)
    for target, t in summary["targets"].items():
        print("")
        print("%s  [%s]   runs=%d  observed-in-runs=%d  longest run=%d"
              % (target, t["unit"], t["n_runs"], t["n_observed_in_runs"], t["longest_run"]))
        print("  population spread: sd=%.1f  median=%.1f" % (t["value_std"], t["value_p50"]))
        print("  difference-based sigma (each an UPPER bound; falling = signal bias shed):")
        for order in sorted(t["difference_estimators"], key=int):
            e = t["difference_estimators"][order]
            print("    order %s : sigma = %8.2f %-5s  (n=%d)"
                  % (order, e["sigma"], t["unit"], e["n_differences"]))
        print("  robust order-1 (MAD)   : sigma = %8.2f %s" % (t["robust_order1_mad_sigma"], t["unit"]))
        for pct, sig in sorted(t["trimmed_order4_sigma"].items(), reverse=True):
            print("  order-4 trimmed at %s%%: sigma = %8.2f %s" % (pct, sig, t["unit"]))
        for label, mass in sorted(t["order4_squared_mass_in_tail"].items()):
            print("  order-4 squared mass carried by %-8s: %5.1f%%" % (label, 100 * mass))
        nug = t["variogram_nugget"]
        print("  variogram nugget       : sigma = %8.2f %s  (gamma = c0 + a*h^%.2f)"
              % (nug["sigma"], t["unit"], nug["exponent"]))
        g1 = t["semivariogram"].get("1")
        if g1:
            print("  one-step semivariogram : gamma(1) = %.0f  -> sqrt = %.2f %s "
                  "(%.1f%% of the population variance)"
                  % (g1, np.sqrt(g1), t["unit"], 100 * g1 / (t["value_std"] ** 2)))

        floor = t["difference_estimators"][max(DIFF_FILTERS)]["sigma"]
        robust = t["robust_order1_mad_sigma"]
        print("  -- what it means for the arms (TEST seed 42 RMSE, same population) --")
        for arm, rmse in LADDER_RMSE[population][target].items():
            frac = (floor / rmse) ** 2 if rmse else float("nan")
            frac_r = (robust / rmse) ** 2 if rmse else float("nan")
            print("    %-12s RMSE=%8.2f %-5s  irreducible share: %5.1f%% (order-4)"
                  "   %5.1f%% (MAD bulk)" % (arm, rmse, t["unit"], 100 * frac, 100 * frac_r))


def main() -> None:
    out_path = Path("data/.noise_floor.json")
    result = {}
    for population, cut_ev in (("cut", 3000.0), ("inclusive", 0.0)):
        bundle = collect(cut_ev)
        summary = summarize(bundle, cut_ev)
        result[population] = summary
        report(summary, population)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("")
    print("wrote %s" % out_path)


if __name__ == "__main__":
    main()
