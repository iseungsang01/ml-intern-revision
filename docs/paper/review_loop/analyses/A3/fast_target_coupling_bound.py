"""Review-loop round 1, analysis A3: noise-corrected bound on the fast-channel /
target association at 10-200 ms.

Question: after correcting for the noisier V_rot label, is the association of the
fast diagnostics (BES / ECEI modality means) with V_rot bounded well below the T_i
association at every time scale 10-200 ms?

Design (real CSVs only, CPU):
  * 641 shot CSVs, held-free targets (dataset/analyze_noise_floor held rule),
    T_i cut at 3000 eV, V_rot no cut.  Blocks split at time gaps >= 0.5 s.
  * Fast modality mean = per-shot z-scored channel mean (analyze_foundations
    .modality_mean_z), BES_* and ECEI_*.
  * For w = k*10 ms, k in {1,2,5,10,20}: target difference D_k y_t = y_t - y_{t-k}
    (both observed, same block, exact k-step spacing); fast difference at lag L
    steps (L = 0..20, 0-200 ms; fast leads): D_k x_{t-L} = x_{t-L} - x_{t-L-k}.
  * Pooled within-shot Pearson r (each variable centred within shot), shot-cluster
    bootstrap 95 % percentile CI, B = 10,000 (seed 0).
  * Noise correction: the w-difference of a white measurement noise has variance
    2 sigma_meas^2; signal fraction f = 1 - 2 sigma^2 / Var(D_k y) (pooled within
    shot); r_corr = r / sqrt(f), computed in each bootstrap replicate.  sigma_meas
    from the same difference estimators the paper quotes (data/.noise_floor.json,
    produced by ces_prediction/analyze_noise_floor.py): robust order-1 MAD (low
    end) and order-4 binomial (high end, the paper's 46.4-129.9 eV range for T_i).
  * Level correlation at each scale: target level y_t vs the w-boxcar mean of the
    fast channel ending at t-L.
Run from repo root:  py docs/paper/review_loop/analyses/A3/fast_target_coupling_bound.py
"""
from __future__ import annotations

import json
import os
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd

CUT_EV = 3000.0
GAP = 0.5
STEP = 0.01
KS = [1, 2, 5, 10, 20]
LAGS = list(range(0, 21))
B = 10_000
OUT_DIR = Path("docs/paper/review_loop/analyses/A3")
OUT_MAIN = Path("docs/paper/fast_target_coupling_bound.json")


def stuck_mask(values, block):
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


def modality_mean_z(df, cols):
    a = df[cols].to_numpy(dtype=float)
    mu = np.nanmean(a, axis=0, keepdims=True)
    sd = np.nanstd(a, axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    return np.nanmean((a - mu) / sd, axis=1)


def lagged(arr, s):
    """out[t] = arr[t - s] (NaN where out of range)."""
    out = np.full(arr.shape, np.nan)
    if s == 0:
        return arr.copy()
    out[s:] = arr[:-s]
    return out


def main():
    t0 = _time.time()
    data_dir = Path(os.environ.get("CES_DATA_DIR", "data"))
    files = sorted(data_dir.glob("*.csv"))
    if len(files) < 600:
        raise SystemExit("expected 641 shot CSVs in %s, found %d" % (data_dir, len(files)))
    nf = json.loads(Path("data/.noise_floor.json").read_text())
    sig = {
        "CES_TI": {"mad": nf["cut"]["targets"]["CES_TI"]["robust_order1_mad_sigma"],
                   "order4": nf["cut"]["targets"]["CES_TI"]["difference_estimators"]["4"]["sigma"],
                   "order1_rice": nf["cut"]["targets"]["CES_TI"]["difference_estimators"]["1"]["sigma"]},
        "CES_VT": {"mad": nf["cut"]["targets"]["CES_VT"]["robust_order1_mad_sigma"],
                   "order4": nf["cut"]["targets"]["CES_VT"]["difference_estimators"]["4"]["sigma"],
                   "order1_rice": nf["cut"]["targets"]["CES_VT"]["difference_estimators"]["1"]["sigma"]},
    }
    head = pd.read_csv(files[0], nrows=1)
    bes_cols = [c for c in head.columns if c.startswith("BES_")]
    ecei_cols = [c for c in head.columns if c.startswith("ECEI_")]
    targets = ["CES_TI", "CES_VT"]
    mods = {"BES": bes_cols, "ECEI": ecei_cols}
    kinds = ["diff", "level"]
    # stats[kind][tgt][mod][k][L] -> per-shot arrays n, sxy, sxx, syy
    S = {kd: {t: {m: {k: {L: np.zeros((len(files), 4)) for L in LAGS} for k in KS} for m in mods} for t in targets} for kd in kinds}

    for fi, path in enumerate(files):
        df = pd.read_csv(path)
        tm = df["time"].to_numpy(dtype=float)
        block = np.concatenate(([0], np.cumsum(np.diff(tm) >= GAP)))
        ys = {}
        for t in targets:
            v = df[t].to_numpy(dtype=float).copy()
            if t == "CES_TI":
                v[v > CUT_EV] = np.nan
            v[stuck_mask(v, block)] = np.nan
            ys[t] = v
        xs = {m: modality_mean_z(df, c) for m, c in mods.items()}
        for k in KS:
            # valid k-step spacing within a block ending at t
            tk = lagged(tm, k)
            bk = lagged(block.astype(float), k)
            okk = (np.abs(tm - tk - k * STEP) < 0.5 * STEP) & (bk == block)
            for t in targets:
                y = ys[t]
                dy = np.where(okk, y - lagged(y, k), np.nan)
                for m in mods:
                    x = xs[m]
                    dx = np.where(okk, x - lagged(x, k), np.nan)
                    # boxcar mean of x over k steps ending at t (same block)
                    cs = pd.Series(x).rolling(k, min_periods=k).mean().to_numpy()
                    okbox = (lagged(block.astype(float), k - 1) == block) if k > 1 else np.ones_like(okk)
                    xbox = np.where(okbox, cs, np.nan)
                    for L in LAGS:
                        bl = lagged(block.astype(float), L)
                        tl = lagged(tm, L)
                        okl = (bl == block) & (np.abs(tm - tl - L * STEP) < 0.5 * STEP)
                        for kd, xx, yy in (("diff", lagged(dx, L), dy), ("level", lagged(xbox, L), y)):
                            sel = okl & np.isfinite(xx) & np.isfinite(yy)
                            n = int(sel.sum())
                            if n < 3:
                                continue
                            a = xx[sel] - xx[sel].mean()
                            b = yy[sel] - yy[sel].mean()
                            S[kd][t][m][k][L][fi] = (n, a @ b, a @ a, b @ b)
        if fi % 100 == 0:
            print("file %d/%d  %.0fs" % (fi, len(files), _time.time() - t0), flush=True)

    rng = np.random.default_rng(0)
    W = rng.multinomial(len(files), np.full(len(files), 1.0 / len(files)), size=B).astype(float)

    def ci(v):
        v = v[np.isfinite(v)]
        if v.size < 0.9 * B:
            return [float("nan"), float("nan")]
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

    res = {kd: {t: {m: {} for m in mods} for t in targets} for kd in kinds}
    for kd in kinds:
        for t in targets:
            for m in mods:
                for k in KS:
                    rows = []
                    for L in LAGS:
                        st = S[kd][t][m][k][L]
                        tot = st.sum(0)
                        r = tot[1] / np.sqrt(tot[2] * tot[3])
                        bt = W @ st
                        rb = bt[:, 1] / np.sqrt(bt[:, 2] * bt[:, 3])
                        row = {"lag_ms": L * 10, "n_rows": int(tot[0]),
                               "n_shots": int((st[:, 0] > 0).sum()),
                               "r": float(r), "ci95": ci(rb)}
                        if kd == "diff":
                            vdy = tot[3] / tot[0]
                            vdy_b = bt[:, 3] / bt[:, 0]
                            row["var_dy"] = float(vdy)
                            for sname in ("mad", "order4", "order1_rice"):
                                nv = 2 * sig[t][sname] ** 2
                                f = 1 - nv / vdy
                                fb = 1 - nv / vdy_b
                                with np.errstate(invalid="ignore", divide="ignore"):
                                    rc = r / np.sqrt(f) if f > 0 else float("nan")
                                    rcb = np.where(fb > 0, rb / np.sqrt(np.where(fb > 0, fb, 1)), np.nan)
                                row["corr_" + sname] = {"signal_fraction": float(f), "r_corr": float(rc), "ci95": ci(rcb)}
                        rows.append(row)
                    res[kd][t][m][str(k * 10)] = rows

    # decision-rule summary
    summ = {}
    for sname in ("mad", "order4"):
        s = {}
        for t in targets:
            for m in mods:
                for k in KS:
                    rows = res["diff"][t][m][str(k * 10)]
                    lag0 = rows[0]["corr_" + sname]
                    allabs = [max(abs(x) for x in rw["corr_" + sname]["ci95"]) for rw in rows]
                    los = [rw["corr_" + sname]["ci95"][0] for rw in rows]
                    s["%s|%s|w=%dms" % (t, m, k * 10)] = {
                        "lag0_r_corr": lag0["r_corr"], "lag0_ci95": lag0["ci95"],
                        "lag0_signal_fraction": lag0["signal_fraction"],
                        "lag0_abs_upper": max(abs(x) for x in lag0["ci95"]),
                        "max_over_lags_abs_upper": float(np.nanmax(allabs)),
                        "best_lag_r_corr": float(np.nanmax([abs(rw["corr_" + sname]["r_corr"]) for rw in rows])),
                        "n_lags_ci_excl_0": int(sum(1 for rw in rows if rw["corr_" + sname]["ci95"][0] > 0 or rw["corr_" + sname]["ci95"][1] < 0)),
                        "max_over_lags_lower": float(np.nanmax(los)),
                    }
        summ[sname] = s
    out = {
        "analysis": "review_loop round 1 A3: noise-corrected fast-channel/target coupling bound",
        "date": "2026-11-13",
        "data": {"n_files": len(files), "cut_ev_TI": CUT_EV, "held": "removed", "gap_s": GAP,
                 "fast": "per-shot z-scored modality mean (BES_*, ECEI_*)",
                 "bes_cols": bes_cols, "ecei_cols": ecei_cols},
        "bootstrap": {"B": B, "unit": "shot", "seed": 0, "ci": "percentile 95%"},
        "sigma_meas": sig,
        "sigma_source": "data/.noise_floor.json (ces_prediction/analyze_noise_floor.py, cut population)",
        "windows_ms": [k * 10 for k in KS], "lags_ms": [L * 10 for L in LAGS],
        "summary": summ,
        "results": res,
        "runtime_s": _time.time() - t0,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    txt = json.dumps(out, indent=1)
    OUT_MAIN.write_text(txt)
    (OUT_DIR / "fast_target_coupling_bound.json").write_text(txt)
    for sname in ("mad", "order4"):
        print("== sigma estimator: %s" % sname)
        for key, v in summ[sname].items():
            print("%-26s r0c=%+.3f [%+.3f,%+.3f] f=%.2f  maxabsUB(lags)=%.3f  lagsExcl0=%d" % (
                key, v["lag0_r_corr"], v["lag0_ci95"][0], v["lag0_ci95"][1], v["lag0_signal_fraction"],
                v["max_over_lags_abs_upper"], v["n_lags_ci_excl_0"]))
    print("raw lag0 diff r:")
    for t in targets:
        for m in mods:
            print(t, m, " ".join("%d:%+.3f" % (k * 10, res["diff"][t][m][str(k * 10)][0]["r"]) for k in KS))
    print("lag0 level r:")
    for t in targets:
        for m in mods:
            print(t, m, " ".join("%d:%+.3f" % (k * 10, res["level"][t][m][str(k * 10)][0]["r"]) for k in KS))
    print("runtime %.0fs" % (_time.time() - t0))


if __name__ == "__main__":
    main()
