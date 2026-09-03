"""Measure the physical time scales the data contains, and test the one term of the
angular-momentum balance that our diagnostics could still reach.

Why this exists
---------------
`docs/paper/main_ko.tex` carries nine tables, seven figures and **zero displayed
equations**, and its model section is 53 lines against 596 for results.  Every time
scale in the paper is therefore empirical and unanchored: the ~50 ms context
saturation (S8al), the 10 ms grid, the aliased Mirnov stream (S8b.2).  This script
supplies the missing physical side, in three clearly separated parts.

PART 1 -- MEASURED time scales, from the 641 shot CSVs under the confirmed data
   treatment: autocorrelation e-folding time of every channel.  If `T_i` and `V_rot`
   relax on the *same* scale, then the difference in predictability is not a response-time
   difference; it is a source-term difference.  That is a testable physical statement
   and it is what the transport analogy (Prandtl number of order one, so momentum and
   heat diffusivities are comparable) predicts.

PART 2 -- MEASURED coupling, in the coordinate the conservation laws actually use.
   The transport equations constrain a *rate*, not a level:

       d/dt (3/2 n_i T_i) = Q_ei(T_e - T_i) - div(q_i) + ...
       d/dt (n_i m_i <R^2> omega) = -div(Pi_phi) + T_NBI + T_NTV + T_int + ...

   So the honest question is not "do the fast diagnostics correlate with `V_rot`" --
   S8ab already answered that with an ablation -- but "do they correlate with
   **dV_rot/dt**".  Reading the torque balance term by term:

     | term            | in our data?                  | status                    |
     | LHS  dL/dt      | yes (finite difference)        | measured here             |
     | T_NBI           | no 0-D channel at all          | absent (S8b.3)            |
     | T_NTV  ~ dB^2   | Mirnov, but 100 Hz decimated   | CLOSED negative (S8b.2):  |
     |                 |                                | |MC| and rolling RMS      |
     |                 |                                | already tried, no help    |
     | div(Pi_turb)    | BES sees density fluctuation   | **never tested**          |
     | T_intrinsic     | needs grad T_i, we have scalar | not reachable             |

   The turbulent momentum flux is the one term our inputs could still carry, and it
   enters the equation as a rate.  This part measures it and does the same for `T_i`
   as the internal control.  A null here is a *stronger* statement than the existing
   data-level null, because it is a null about a named term of a conservation law.

PART 3 -- THEORY: the electron-ion energy equipartition time from the standard
   Braginskii/NRL electron collision time, over a RANGE of KSTAR-like (n_e, T_e).
   Those parameters are NOT in our CSVs -- BES and ECEI are in instrument units, not
   m^-3 and eV -- so they are quoted assumptions and are printed as such.  The claim
   is order-of-magnitude and the factor-of-2 convention in tau_eq is stated, not hidden.

No model is trained, scored, selected or re-selected here.  This is a data-and-theory
audit in the same class as `analyze_data_evidence.py` and `analyze_noise_floor.py`.

Run from the repo root:  py ces_prediction/analyze_physics_scales.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

STUCK_GAP_SECONDS = 0.5
CUT_EV = 3000.0
MAX_LAG = 30          # +/- 300 ms on the 10 ms grid
MIN_OBS = 40          # observed target rows a block needs to contribute

# --------------------------------------------------------------------------
# PART 3 constants -- ASSUMED plasma parameters.  NOT measured by this repo.
# --------------------------------------------------------------------------
NE_RANGE_M3 = [2.0e19, 3.0e19, 5.0e19]
TE_RANGE_EV = [500.0, 1000.0, 2000.0, 3000.0]
LN_LAMBDA = 17.0
Z_EFF = 1.0
MI_OVER_ME_D = 3671.5     # deuterium


def tau_e_seconds(te_ev, ne_m3, lnlam=LN_LAMBDA, z=Z_EFF):
    """Braginskii electron collision time, NRL practical form.

        tau_e [s] = 3.44e5 * Te[eV]^1.5 / (Z * ne[cm^-3] * lnLambda)
    """
    return 3.44e5 * te_ev**1.5 / (z * (ne_m3 * 1e-6) * lnlam)


def tau_equipartition_seconds(te_ev, ne_m3):
    """Electron-ion temperature equilibration time, tau_eq = (m_i / 2 m_e) * tau_e.

    Texts carry an order-unity factor in front of this; the comparison it supports is
    an order-of-magnitude one and is reported as such.
    """
    return (MI_OVER_ME_D / 2.0) * tau_e_seconds(te_ev, ne_m3)


# --------------------------------------------------------------------------
# data handling -- mirrors dataset.py
# --------------------------------------------------------------------------
def data_dir():
    return Path(os.environ.get("CES_DATA_DIR", "data"))


def _stuck_mask(values, block):
    stuck = np.zeros(values.shape, dtype=bool)
    last_val, last_block = np.nan, -1
    for i, v in enumerate(values):
        if np.isnan(v):
            continue
        if block[i] == last_block and not np.isnan(last_val) and v == last_val:
            stuck[i] = True
        else:
            last_val, last_block = v, block[i]
    return stuck


def clean_target(df, col, block):
    values = df[col].to_numpy(dtype=float).copy()
    if col == "CES_TI":
        values[values > CUT_EV] = np.nan
    values[_stuck_mask(values, block)] = np.nan
    return values


def modality_mean(df, cols):
    """Per-shot standardized mean over a modality's channels (S8s does the same)."""
    if not cols:
        return None
    a = df[cols].to_numpy(dtype=float)
    mu = np.nanmean(a, axis=0, keepdims=True)
    sd = np.nanstd(a, axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    return np.nanmean((a - mu) / sd, axis=1)


def one_step_increment(values):
    """d(value)/dt across CONSECUTIVE observed pairs only; NaN elsewhere.

    Placed at the later of the two rows, so it is the change that arrived by t.
    """
    d = np.full(values.shape, np.nan)
    ok = ~np.isnan(values)
    pair = ok[1:] & ok[:-1]
    d[1:][pair] = values[1:][pair] - values[:-1][pair]
    return d


class Pooled:
    """Accumulates a pooled Pearson r over blocks and files, per lag."""

    def __init__(self, n_lags):
        self.num = np.zeros(n_lags)
        self.da = np.zeros(n_lags)
        self.db = np.zeros(n_lags)

    def add(self, j, a, b):
        self.num[j] += float(np.sum(a * b))
        self.da[j] += float(np.sum(a * a))
        self.db[j] += float(np.sum(b * b))

    def r(self):
        with np.errstate(invalid="ignore", divide="ignore"):
            return self.num / np.sqrt(self.da * self.db)


def efolding_ms(r, step_ms):
    thr = 1.0 / np.e
    for k in range(1, len(r)):
        if not np.isfinite(r[k]):
            return float("nan")
        if r[k] < thr:
            prev = r[k - 1]
            if prev == r[k]:
                return k * step_ms
            return (k - 1 + (prev - thr) / (prev - r[k])) * step_ms
    return float("nan")


def main():
    files = sorted(data_dir().glob("*.csv"))
    if not files:
        raise SystemExit("no shot CSVs found -- point CES_DATA_DIR at the real folder")

    head = pd.read_csv(files[0], nrows=1)
    groups = {
        "BES": [c for c in head.columns if c.startswith("BES_")],
        "ECEI": [c for c in head.columns if c.startswith("ECEI_")],
        "MC": [c for c in head.columns if c.startswith("MC")],
    }
    chans = ["CES_TI", "CES_VT", "BES", "ECEI", "MC"]
    lags = np.arange(-MAX_LAG, MAX_LAG + 1)

    auto = {c: Pooled(MAX_LAG + 1) for c in chans}
    # Held removal preferentially deletes rows whose value did NOT change, so it
    # biases a V_rot autocorrelation DOWN; keeping held biases it UP (a repeat is a
    # perfect correlation the instrument never measured).  Report both: the truth is
    # bracketed, and only V_rot is affected (T_i has 1 held row in 641 files).
    auto_kept = {c: Pooled(MAX_LAG + 1) for c in ("CES_TI", "CES_VT")}
    cross = {}
    for drv in ("BES", "ECEI", "MC"):
        for tgt in ("CES_TI", "CES_VT"):
            for form in ("level", "rate"):
                cross[(drv, tgt, form)] = Pooled(lags.size)

    step = None
    n_blocks = 0

    for path in files:
        df = pd.read_csv(path)
        time = df["time"].to_numpy(dtype=float)
        if step is None and len(time) > 2:
            step = float(np.median(np.diff(time)))
        block = np.concatenate(([0], np.cumsum(np.diff(time) >= STUCK_GAP_SECONDS)))

        series = {c: modality_mean(df, groups[c]) for c in groups}
        for c in ("CES_TI", "CES_VT"):
            series[c] = clean_target(df, c, block)
        rate = {c: one_step_increment(series[c]) for c in ("CES_TI", "CES_VT")}
        kept = {}
        for c in ("CES_TI", "CES_VT"):
            v = df[c].to_numpy(dtype=float).copy()
            if c == "CES_TI":
                v[v > CUT_EV] = np.nan
            kept[c] = v

        for b in np.unique(block):
            sel = block == b
            if sel.sum() < MIN_OBS:
                continue
            n_blocks += 1
            loc = {k: (v[sel] - np.nanmean(v[sel])) for k, v in series.items()}
            loc_rate = {k: (v[sel] - np.nanmean(v[sel])) for k, v in rate.items()}

            for c in chans:
                s = loc[c]
                if np.isfinite(s).sum() < MIN_OBS:
                    continue
                for k in range(MAX_LAG + 1):
                    a = s[:len(s) - k] if k else s
                    cser = s[k:]
                    m = np.isfinite(a) & np.isfinite(cser)
                    if m.sum() >= 10:
                        auto[c].add(k, a[m], cser[m])

            for c in ("CES_TI", "CES_VT"):
                s_k = kept[c][sel]
                s_k = s_k - np.nanmean(s_k)
                if np.isfinite(s_k).sum() < MIN_OBS:
                    continue
                for k in range(MAX_LAG + 1):
                    a = s_k[:len(s_k) - k] if k else s_k
                    cser = s_k[k:]
                    m = np.isfinite(a) & np.isfinite(cser)
                    if m.sum() >= 10:
                        auto_kept[c].add(k, a[m], cser[m])

            for drv in ("BES", "ECEI", "MC"):
                d = loc[drv]
                if not np.isfinite(d).any():
                    continue
                for tgt in ("CES_TI", "CES_VT"):
                    for form, tser in (("level", loc[tgt]), ("rate", loc_rate[tgt])):
                        if np.isfinite(tser).sum() < MIN_OBS:
                            continue
                        n = len(tser)
                        acc = cross[(drv, tgt, form)]
                        for j, k in enumerate(lags):
                            if k >= 0:
                                a, c2 = d[:n - k], tser[k:]
                            else:
                                a, c2 = d[-k:], tser[:n + k]
                            m = np.isfinite(a) & np.isfinite(c2)
                            if m.sum() >= 10:
                                acc.add(j, a[m], c2[m])

    step_ms = step * 1000.0
    out = {
        "n_files": len(files), "n_blocks": n_blocks,
        "grid_step_ms": round(step_ms, 3), "nyquist_hz": round(0.5 / step, 3),
        "part1_autocorrelation": {}, "part2_coupling": {}, "part3_theory": {},
        "assumptions": {
            "note": "plasma parameters are NOT measured by this repo (BES/ECEI are in "
                    "instrument units). Quoted for the KSTAR 2022 campaign range.",
            "ne_m3": NE_RANGE_M3, "te_eV": TE_RANGE_EV,
            "ln_lambda": LN_LAMBDA, "z_eff": Z_EFF, "ion": "deuterium",
            "tau_eq_convention": "tau_eq = (m_i / 2 m_e) * tau_e; order-of-magnitude",
        },
    }

    print("=" * 76)
    print("PART 1 -- MEASURED relaxation scales (641 files, held-free, cut, %d blocks)"
          % n_blocks)
    print("=" * 76)
    print("%-8s %9s %9s %9s %9s %11s" % ("channel", "r(10ms)", "r(50ms)", "r(100ms)",
                                         "r(300ms)", "1/e time"))
    for c in chans:
        r = auto[c].r()
        tau = efolding_ms(r, step_ms)
        print("%-8s %9.3f %9.3f %9.3f %9.3f %8.0f ms"
              % (c, r[1], r[5], r[10], r[30], tau))
        out["part1_autocorrelation"][c] = {
            "r": [float(x) for x in r], "efolding_ms": float(tau)}
    print("")
    print("  held-KEPT sensitivity (upper bound; a repeat is a correlation never measured):")
    for c in ("CES_TI", "CES_VT"):
        r = auto_kept[c].r()
        tau = efolding_ms(r, step_ms)
        print("  %-6s %9.3f %9.3f %9.3f %9.3f %8.0f ms"
              % (c, r[1], r[5], r[10], r[30], tau))
        out["part1_autocorrelation"][c + "_held_kept"] = {
            "r": [float(x) for x in r], "efolding_ms": float(tau)}

    print("")
    print("=" * 76)
    print("PART 2 -- MEASURED coupling in the conservation-law coordinate")
    print("  level = fast channel vs the target itself")
    print("  rate  = fast channel vs the one-step increment (what the equation sets)")
    print("  k > 0 means the fast channel LEADS the target by k")
    print("=" * 76)
    print("%-18s %10s %10s %12s" % ("pair", "peak |r|", "peak lag", "r at lag 0"))
    for (drv, tgt, form), acc in cross.items():
        r = acc.r()
        if not np.isfinite(r).any():
            continue
        j = int(np.nanargmax(np.abs(r)))
        z = int(np.where(lags == 0)[0][0])
        name = "%s->%s(%s)" % (drv, tgt, form)
        print("%-18s %10.3f %8.0f ms %12.3f" % (name, r[j], lags[j] * step_ms, r[z]))
        out["part2_coupling"][name] = {
            "lags_ms": [float(l * step_ms) for l in lags],
            "r": [float(x) for x in r],
            "peak_abs_r": float(r[j]), "peak_lag_ms": float(lags[j] * step_ms),
            "r_at_zero": float(r[z]),
        }

    print("")
    print("=" * 76)
    print("PART 3 -- THEORY: electron-ion equipartition (ASSUMED plasma parameters)")
    print("  tau_e  = 3.44e5 * Te^1.5 / (Z * ne[cm^-3] * lnLambda)   [Braginskii/NRL]")
    print("  tau_eq = (m_i / 2 m_e) * tau_e,  deuterium, lnLambda = %.0f, Z = %.0f"
          % (LN_LAMBDA, Z_EFF))
    print("=" * 76)
    print("%-10s" % "Te \\ ne" + "".join("%16s" % ("%.0e m^-3" % n) for n in NE_RANGE_M3))
    theory = {}
    for te in TE_RANGE_EV:
        row = "%-10s" % ("%.0f eV" % te)
        theory[str(int(te))] = {}
        for ne in NE_RANGE_M3:
            t = tau_equipartition_seconds(te, ne) * 1000.0
            row += "%13.1f ms" % t
            theory[str(int(te))]["%.0e" % ne] = float(t)
        print(row)
    out["part3_theory"]["tau_eq_ms"] = theory
    out["part3_theory"]["tau_e_us_at_1keV_3e19"] = tau_e_seconds(1000.0, 3.0e19) * 1e6

    print("")
    print("Grid %.0f ms -> Nyquist %.0f Hz. Mirnov mode rotation is kHz, so the decimated"
          % (step_ms, 0.5 / step))
    print("MC stream cannot represent it -- a sampling statement, not a model one, and it")
    print("is why S8b.2 measures lag-1 autocorrelation at -0.009 against BES +0.568.")

    Path("data").mkdir(exist_ok=True)
    Path("data/.physics_scales.json").write_text(json.dumps(out, indent=2),
                                                 encoding="utf-8")
    print("")
    print("wrote data/.physics_scales.json")


if __name__ == "__main__":
    main()
