"""P06 -- Gaussian MI / partial-correlation bound on what the fast diagnostics can add
to a causal T_i nowcast, checked on the raw 641 CSVs.

Protocol (same as analyze_foundations.py Part D / analyze_physics_scales.py):
  held-free (repeat of previous observed value inside a block -> NaN), T_i > 3 keV -> NaN,
  blocks split at time gaps >= 0.5 s, consecutive observed grid pairs only.
Innovation proxy:  e_t = Ti(t) - [m + rho (Ti(t-1) - m)],  rho = exp(-10/81), m = block mean.
Regressors: 15 fast channels (per-shot z-scored, as CES_PER_SHOT_NORM=1 and S8ar) at t,
            and their one-step increments z(t) - z(t-1)  -> 30 regressors.
All pooled statistics are within-block centred.  No model is trained or scored.
ASCII-only output (cp949 console).
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\lss\Documents\GitHub\ml-intern-revision")
DATA = Path(os.environ.get("CES_DATA_DIR", ROOT / "data"))
OUT_DIR = Path(__file__).resolve().parent
STUCK_GAP_SECONDS = 0.5
CUT_EV = 3000.0
DT = 0.010
TAU_MS = 81.18883695943607          # data/.foundations.json D.CES_TI.fits.robust.tau_ms
RHO = float(np.exp(-10.0 / TAU_MS))
MIN_OBS = 40
TAIL_FRAC = 0.10


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


def per_shot_z(a):
    mu = np.nanmean(a, axis=0, keepdims=True)
    sd = np.nanstd(a, axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    return (a - mu) / sd


def r2_of_fit(X, y, w=None, Xte=None, yte=None, lam=1e-8):
    """OLS with intercept (ridge with tiny lambda for conditioning). Returns in-sample R^2
    and, if a test set is given, out-of-sample R^2 (1 - SSE/SST on the test set, SST about
    the test mean)."""
    X1 = np.column_stack([np.ones(len(X)), X])
    A = X1.T @ X1 + lam * np.eye(X1.shape[1])
    beta = np.linalg.solve(A, X1.T @ y)
    res = y - X1 @ beta
    r2 = 1.0 - res @ res / ((y - y.mean()) @ (y - y.mean()))
    if Xte is None:
        return r2, beta
    X1te = np.column_stack([np.ones(len(Xte)), Xte])
    rte = yte - X1te @ beta
    r2te = 1.0 - rte @ rte / ((yte - yte.mean()) @ (yte - yte.mean()))
    return r2, r2te, beta


def main():
    files = sorted(DATA.glob("*.csv"))
    if not files:
        raise SystemExit("no shot CSVs found -- point CES_DATA_DIR at the real folder")
    head = pd.read_csv(files[0], nrows=1)
    bes = [c for c in head.columns if c.startswith("BES_")]
    ecei = [c for c in head.columns if c.startswith("ECEI_")]
    mc = [c for c in head.columns if c.startswith("MC")]
    fast = bes + ecei + mc
    print("files", len(files), "fast channels", len(fast), "rho", round(RHO, 4))

    rows_e, rows_z, rows_dz, rows_mod, rows_shot, rows_block = [], [], [], [], [], []
    rows_prev = []
    rows_lag = []
    rows_hist = []
    rows_pers = []
    n_blocks = 0
    for fi, path in enumerate(files):
        df = pd.read_csv(path)
        time = df["time"].to_numpy(dtype=float)
        block = np.concatenate(([0], np.cumsum(np.diff(time) >= STUCK_GAP_SECONDS)))
        ti = df["CES_TI"].to_numpy(dtype=float).copy()
        ti[ti > CUT_EV] = np.nan
        ti[_stuck_mask(ti, block)] = np.nan
        Z = per_shot_z(df[fast].to_numpy(dtype=float))
        MOD = np.column_stack([np.nanmean(Z[:, :len(bes)], axis=1),
                               np.nanmean(Z[:, len(bes):len(bes) + len(ecei)], axis=1),
                               np.nanmean(Z[:, len(bes) + len(ecei):], axis=1)])
        for b in np.unique(block):
            sel = np.flatnonzero(block == b)
            tb, yb = time[sel], ti[sel]
            if np.isfinite(yb).sum() < MIN_OBS:
                continue
            m = np.nanmean(yb)
            ok = np.isfinite(yb[1:]) & np.isfinite(yb[:-1]) & (np.abs((tb[1:] - tb[:-1]) - DT) < 0.5 * DT)
            idx = np.flatnonzero(ok) + 1              # row t (later of the pair) inside block
            if idx.size < 10:
                continue
            e = yb[idx] - (m + RHO * (yb[idx - 1] - m))
            Zb = Z[sel]
            z_t, z_tm1 = Zb[idx], Zb[idx - 1]
            LAGS = (2, 3, 5)
            idx_l = np.clip(idx[:, None] - np.array(LAGS)[None, :], 0, None)       # (n, 3)
            z_lag = np.stack([Zb[idx_l[:, k]] for k in range(len(LAGS))], axis=1)   # (n, 3, 15)
            lag_ok = (idx[:, None] - np.array(LAGS)[None, :] >= 0).all(axis=1) & np.isfinite(z_lag).all(axis=(1, 2))
            good = np.isfinite(z_t).all(axis=1) & np.isfinite(z_tm1).all(axis=1) & lag_ok
            if good.sum() < 10:
                continue
            e, z_t, z_tm1 = e[good], z_t[good], z_tm1[good]
            z_lag = z_lag[good].reshape(good.sum(), -1)
            mod_t = MOD[sel][idx][good]
            dz = z_t - z_tm1
            z_lag = z_lag - z_lag.mean(axis=0)
            rows_lag.append(z_lag)
            # within-block centring of every variable
            e = e - e.mean()
            z_t = z_t - z_t.mean(axis=0)
            dz = dz - dz.mean(axis=0)
            mod_t = mod_t - mod_t.mean(axis=0)
            rows_e.append(e); rows_z.append(z_t); rows_dz.append(dz); rows_mod.append(mod_t)
            rows_shot.append(np.full(e.size, fi)); rows_block.append(np.full(e.size, n_blocks))
            rows_prev.append(yb[idx - 1][good] - m)
            HL = (2, 3, 5)
            idx_h = np.clip(idx[:, None] - np.array(HL)[None, :], 0, None)
            hl = np.stack([yb[idx_h[:, k]] for k in range(len(HL))], axis=1) - m
            hl[(idx[:, None] - np.array(HL)[None, :]) < 0] = np.nan
            rows_hist.append(hl[good])
            rows_pers.append((yb[idx] - yb[idx - 1])[good])
            n_blocks += 1

    e = np.concatenate(rows_e); Zt = np.vstack(rows_z); dZ = np.vstack(rows_dz)
    MOD = np.vstack(rows_mod); shot = np.concatenate(rows_shot)
    prev = np.concatenate(rows_prev)
    ZL = np.vstack(rows_lag)
    HIST = np.vstack(rows_hist)
    PERS = np.concatenate(rows_pers)
    n = e.size
    print("blocks", n_blocks, "pooled rows", n, "shots", np.unique(shot).size)
    var_e = float(e @ e / n)
    print("Var(e) pooled [eV^2] =", round(var_e, 1), " sd =", round(np.sqrt(var_e), 1))
    # sanity: residual correlation of e with the previous level (should be ~0 if rho is right)
    r_prev = float(np.corrcoef(e, prev)[0, 1])
    print("corr(e, Ti(t-1)-m) =", round(r_prev, 4), " (AR(1) residual check)")

    out = {"rho": RHO, "n_rows": int(n), "n_blocks": int(n_blocks), "var_e": var_e,
           "corr_e_prev_level": r_prev}

    # ---- (a) single regressors ----
    names_mod = ["BES_mean", "ECEI_mean", "MC_mean"]
    single = {}
    for j, nm in enumerate(names_mod):
        r = float(np.corrcoef(e, MOD[:, j])[0, 1])
        rd = float(np.corrcoef(e, np.diff(np.column_stack([np.zeros(n), MOD[:, j]]), axis=1)[:, 0])[0, 1]) if False else None
        single[nm] = {"r": r, "r2": r * r}
    for j, nm in enumerate(fast):
        r = float(np.corrcoef(e, Zt[:, j])[0, 1])
        rinc = float(np.corrcoef(e, dZ[:, j])[0, 1])
        single[nm] = {"r": r, "r2": r * r, "r_inc": rinc, "r2_inc": rinc * rinc}
    # modality-mean increments
    for j, nm in enumerate(names_mod):
        cols = {0: slice(0, len(bes)), 1: slice(len(bes), len(bes) + len(ecei)), 2: slice(len(bes) + len(ecei), None)}[j]
        dm = dZ[:, cols].mean(axis=1)
        rinc = float(np.corrcoef(e, dm)[0, 1])
        single[nm]["r_inc"] = rinc; single[nm]["r2_inc"] = rinc * rinc
    out["single"] = single
    print("\n(a) single-regressor r / r^2 of e_t (innovation proxy):")
    for nm in names_mod + fast:
        s = single[nm]
        print("  %-10s level r=%+.4f r2=%.5f   increment r=%+.4f r2=%.5f" % (nm, s["r"], s["r2"], s["r_inc"], s["r2_inc"]))

    # ---- (b) multiple R^2 ----
    X15 = Zt
    X30 = np.hstack([Zt, dZ])
    X3 = MOD
    odd = (shot % 2) == 1
    multi = {}
    for nm, X in (("mod3", X3), ("ch15", X15), ("ch15+inc15", X30)):
        r2_in, _ = r2_of_fit(X, e)
        _, r2_oe, _ = r2_of_fit(X[odd], e[odd], Xte=X[~odd], yte=e[~odd])   # fit odd, test even
        _, r2_eo, _ = r2_of_fit(X[~odd], e[~odd], Xte=X[odd], yte=e[odd])   # fit even, test odd
        multi[nm] = {"r2_in": float(r2_in), "r2_oos_fit_odd_test_even": float(r2_oe),
                     "r2_oos_fit_even_test_odd": float(r2_eo),
                     "r2_oos_mean": float(0.5 * (r2_oe + r2_eo)), "p": int(X.shape[1])}
    out["multi"] = multi
    print("\n(b) multiple R^2 of e_t (in-sample / 2-fold by shot-index parity):")
    for nm, s in multi.items():
        print("  %-12s p=%2d  R2_in=%.5f  R2_oos(odd->even)=%.5f  R2_oos(even->odd)=%.5f  mean=%.5f"
              % (nm, s["p"], s["r2_in"], s["r2_oos_fit_odd_test_even"], s["r2_oos_fit_even_test_odd"], s["r2_oos_mean"]))

    # ---- (c) tail (top 10 % |e|) vs rest ----
    thr = float(np.quantile(np.abs(e), 1.0 - TAIL_FRAC))
    tail = np.abs(e) >= thr
    sse_share_tail = float((e[tail] @ e[tail]) / (e @ e))
    print("\n(c) tail threshold |e| >= %.1f eV ; tail rows %d (%.1f%%) carry %.3f of SSE(e)"
          % (thr, tail.sum(), 100 * tail.mean(), sse_share_tail))
    cond = {"threshold_ev": thr, "tail_rows": int(tail.sum()), "sse_share_tail": sse_share_tail}
    for nm, X in (("ch15", X15), ("ch15+inc15", X30)):
        # joint fit, explained share inside each subset
        _, beta = r2_of_fit(X, e)
        X1 = np.column_stack([np.ones(n), X])
        res = e - X1 @ beta
        within = {}
        for lab, msk in (("tail", tail), ("rest", ~tail)):
            sst = e[msk] @ e[msk]       # about pooled zero mean (within-block centred)
            within[lab] = {"explained_share_joint_fit": float(1 - res[msk] @ res[msk] / sst)}
            # separate fit within the subset, in-sample and 2-fold by shot parity
            r2_in, _ = r2_of_fit(X[msk], e[msk])
            a, b = msk & odd, msk & ~odd
            _, r2_oe, _ = r2_of_fit(X[a], e[a], Xte=X[b], yte=e[b])
            _, r2_eo, _ = r2_of_fit(X[b], e[b], Xte=X[a], yte=e[a])
            within[lab].update({"r2_in_subset_fit": float(r2_in), "r2_oos_mean_subset_fit": float(0.5 * (r2_oe + r2_eo)),
                                "r2_oos_odd_even": float(r2_oe), "r2_oos_even_odd": float(r2_eo), "n": int(msk.sum())})
        cond[nm] = within
        for lab in ("tail", "rest"):
            w = within[lab]
            print("  %-12s %-4s n=%6d  joint-fit explained share=%.5f  subset fit R2_in=%.5f  R2_oos=%.5f (%.5f / %.5f)"
                  % (nm, lab, w["n"], w["explained_share_joint_fit"], w["r2_in_subset_fit"], w["r2_oos_mean_subset_fit"],
                     w["r2_oos_odd_even"], w["r2_oos_even_odd"]))
    out["tail"] = cond

    # ---- (b') partial R^2: history h = previous observed level (within-block centred) ----
    # null model e ~ prev ; full model e ~ prev + z.  R2_partial = 1 - SSE_full/SSE_null.
    print("")
    print("(b') partial R^2 of e_t given the previous level (regression-diluted AR(1) coefficient refit):")
    H = prev[:, None]
    _, beta0 = r2_of_fit(H, e)
    print("  fitted coefficient on (Ti(t-1)-m) in the null model: %+.4f  (=> effective rho %.3f vs proxy rho %.3f)"
          % (beta0[1], RHO + beta0[1], RHO))
    partial = {}
    X75 = np.hstack([Zt, dZ, ZL])      # lags 0,1(as increment),2,3,5 -> 75 regressors, 50 ms of fast context
    for nm, X in (("mod3", X3), ("ch15", X15), ("ch15+inc15", X30), ("ch15 lags0-5(75)", X75)):
        def pr2(tr, te):
            _, _, b0 = r2_of_fit(H[tr], e[tr], Xte=H[te], yte=e[te])
            _, _, b1 = r2_of_fit(np.hstack([H[tr], X[tr]]), e[tr], Xte=np.hstack([H[te], X[te]]), yte=e[te])
            r0 = e[te] - np.column_stack([np.ones(te.sum()), H[te]]) @ b0
            r1 = e[te] - np.column_stack([np.ones(te.sum()), H[te], X[te]]) @ b1
            return float(1 - r1 @ r1 / (r0 @ r0))
        allm = np.ones(n, bool)
        p_in = pr2(allm, allm); p_oe = pr2(odd, ~odd); p_eo = pr2(~odd, odd)
        partial[nm] = {"r2_partial_in": p_in, "r2_partial_oos_mean": 0.5 * (p_oe + p_eo), "p": int(X.shape[1])}
        print("  %-18s p=%2d  R2_partial_in=%.5f  oos=%.5f (%.5f / %.5f)" % (nm, X.shape[1], p_in, 0.5 * (p_oe + p_eo), p_oe, p_eo))
    out["partial_given_prev_level"] = partial
    # plain (non-partial) R2 for the 75-regressor set too
    r2_in, _ = r2_of_fit(X75, e)
    _, r2_oe, _ = r2_of_fit(X75[odd], e[odd], Xte=X75[~odd], yte=e[~odd])
    _, r2_eo, _ = r2_of_fit(X75[~odd], e[~odd], Xte=X75[odd], yte=e[odd])
    multi["ch15 lags0-5(75)"] = {"r2_in": float(r2_in), "r2_oos_fit_odd_test_even": float(r2_oe),
                                 "r2_oos_fit_even_test_odd": float(r2_eo), "r2_oos_mean": float(0.5 * (r2_oe + r2_eo)), "p": 75}
    print("  plain R2, 75 regressors: in=%.5f oos=%.5f" % (r2_in, 0.5 * (r2_oe + r2_eo)))

    # ---- (b'') partial R^2 given a DEEPER history (prev levels at lags 1,2,3,5 = 50 ms), and an
    #      EMPIRICAL conversion: skill vs persistence measured on the same pooled rows ----
    print("")
    print("(b'') history = previous levels at lags 1,2,3,5 (rows with all four observed); empirical persistence MSE on the same rows")
    hmask = np.isfinite(HIST).all(axis=1)
    H5 = np.column_stack([prev, HIST])[hmask]
    e5, odd5 = e[hmask], odd[hmask]
    mse_pers5 = float(PERS[hmask] @ PERS[hmask] / hmask.sum())
    deep = {"n": int(hmask.sum()), "mse_pers_same_rows": mse_pers5}
    print("  rows %d  persistence MSE on these rows = %.0f eV^2 (RMSE %.1f)" % (hmask.sum(), mse_pers5, np.sqrt(mse_pers5)))
    for nm, X in (("none", None), ("mod3", X3), ("ch15", X15), ("ch15+inc15", X30), ("ch15 lags0-5(75)", X75)):
        Xf = H5 if X is None else np.hstack([H5, X[hmask]])
        def sse(tr, te):
            _, _, b = r2_of_fit(Xf[tr], e5[tr], Xte=Xf[te], yte=e5[te])
            r = e5[te] - np.column_stack([np.ones(te.sum()), Xf[te]]) @ b
            return float(r @ r), int(te.sum())
        s_oe, n_e = sse(odd5, ~odd5); s_eo, n_o = sse(~odd5, odd5)
        mse_oos = (s_oe + s_eo) / (n_e + n_o)
        allm = np.ones(e5.size, bool); s_in, _ = sse(allm, allm)
        mse_in = s_in / e5.size
        deep[nm] = {"mse_in": mse_in, "mse_oos": mse_oos, "skill_vs_pers_in": 1 - mse_in / mse_pers5,
                    "skill_vs_pers_oos": 1 - mse_oos / mse_pers5}
        print("  h5 + %-18s MSE_in=%.0f MSE_oos=%.0f  skill vs pers (in/oos) = %.4f / %.4f"
              % (nm, mse_in, mse_oos, deep[nm]["skill_vs_pers_in"], deep[nm]["skill_vs_pers_oos"]))
    for nm in ("mod3", "ch15", "ch15+inc15", "ch15 lags0-5(75)"):
        d_in = deep[nm]["skill_vs_pers_in"] - deep["none"]["skill_vs_pers_in"]
        d_oos = deep[nm]["skill_vs_pers_oos"] - deep["none"]["skill_vs_pers_oos"]
        pr_in = 1 - deep[nm]["mse_in"] / deep["none"]["mse_in"]; pr_oos = 1 - deep[nm]["mse_oos"] / deep["none"]["mse_oos"]
        deep[nm].update({"dskill_vs_pers_in": d_in, "dskill_vs_pers_oos": d_oos, "r2_partial_in": pr_in, "r2_partial_oos": pr_oos})
        print("  fast adds over h5: %-18s partial R2 in/oos = %.4f / %.4f  -> dskill vs pers (empirical) = %+.4f / %+.4f"
              % (nm, pr_in, pr_oos, d_in, d_oos))
    # tail decomposition of the h5-conditioned gain (joint fit, explained SSE share inside top-10% |e| rows)
    tail5 = tail[hmask]
    for nm, X in (("ch15+inc15", X30), ("ch15 lags0-5(75)", X75)):
        X0 = np.column_stack([np.ones(e5.size), H5]); X1 = np.column_stack([X0, X[hmask]])
        b0 = np.linalg.solve(X0.T @ X0 + 1e-8 * np.eye(X0.shape[1]), X0.T @ e5)
        b1 = np.linalg.solve(X1.T @ X1 + 1e-8 * np.eye(X1.shape[1]), X1.T @ e5)
        r0, r1 = e5 - X0 @ b0, e5 - X1 @ b1
        tot = r0 @ r0
        gain_tail = float((r0[tail5] @ r0[tail5] - r1[tail5] @ r1[tail5]) / (e5.size * mse_pers5))
        gain_rest = float((r0[~tail5] @ r0[~tail5] - r1[~tail5] @ r1[~tail5]) / (e5.size * mse_pers5))
        share_tail_null = float(r0[tail5] @ r0[tail5] / tot)
        deep[nm]["tail_decomp"] = {"dskill_tail": gain_tail, "dskill_rest": gain_rest, "null_sse_share_tail": share_tail_null,
                                   "r2_within_tail": float(1 - r1[tail5] @ r1[tail5] / (r0[tail5] @ r0[tail5])),
                                   "r2_within_rest": float(1 - r1[~tail5] @ r1[~tail5] / (r0[~tail5] @ r0[~tail5]))}
        print("  tail decomposition (h5 + %s): null SSE share in top-10%% |e| = %.3f ; dskill tail %+.4f + rest %+.4f ; R2 within tail %.4f / rest %.4f"
              % (nm, share_tail_null, gain_tail, gain_rest, deep[nm]["tail_decomp"]["r2_within_tail"], deep[nm]["tail_decomp"]["r2_within_rest"]))
    out["deep_history_h5"] = deep

    # ---- conversion to skill-vs-persistence units ----
    F = json.load(open(ROOT / "data" / ".foundations.json", encoding="utf-8"))
    conv = {}
    for fit in ("robust", "classical"):
        p = F["D"]["CES_TI"]["fits"][fit]
        pr = p["pred"]
        s_n2 = p["sigma_n"] ** 2; s_s2 = p["sigma_s"] ** 2; rho = pr["rho_1step"]
        ratio = pr["mse_filter_inf"] / pr["mse_persistence"]          # MSE_h / MSE_pers
        P_minus = pr["mse_filter_inf"] - s_n2
        var_e_theory = s_s2 * (1 - rho ** 2) + s_n2 * (1 + rho ** 2)    # proxy e_t variance under the OU fit
        conv[fit] = {"mse_h_over_mse_pers": ratio, "skill_filter_vs_pers": pr["skill_filter_vs_pers"],
                     "kalman_gain_ceiling_R2_on_true_innovation": P_minus / pr["mse_filter_inf"],
                     "ceiling_dskill_vs_pers": (P_minus / pr["mse_filter_inf"]) * ratio,
                     "var_e_proxy_theory": var_e_theory,
                     "ceiling_R2_on_proxy": 1 - s_n2 / var_e_theory,
                     "sigma_n2": s_n2}
    meas = F["E"]["ladder_1step_CES_TI"]
    gain_meas = meas["model"]["skill_vs_pers"] - meas["gp_causal"]["skill_vs_pers"]
    ratio_meas = 1 - meas["gp_causal"]["skill_vs_pers"]                  # measured MSE_h/MSE_pers (causal GP)
    conv["measured"] = {"skill_model": meas["model"]["skill_vs_pers"], "skill_gp_causal": meas["gp_causal"]["skill_vs_pers"],
                        "gain": gain_meas, "mse_h_over_mse_pers_gp_causal": ratio_meas,
                        "implied_R2_partial": gain_meas / ratio_meas,
                        "implied_r_partial": float(np.sqrt(gain_meas / ratio_meas)),
                        "implied_MI_nats": float(-0.5 * np.log(1 - gain_meas / ratio_meas))}
    out["conversion"] = conv
    print("\nconversion: dskill_vs_pers = R2 * MSE_h/MSE_pers")
    for fit in ("robust", "classical"):
        c = conv[fit]
        print("  %-9s MSE_h/MSE_pers=%.3f  skill_h=%.3f  Kalman ceiling R2=%.3f -> dskill<=%.3f ; proxy var theory=%.0f ceiling R2(proxy)=%.3f"
              % (fit, c["mse_h_over_mse_pers"], c["skill_filter_vs_pers"], c["kalman_gain_ceiling_R2_on_true_innovation"],
                 c["ceiling_dskill_vs_pers"], c["var_e_proxy_theory"], c["ceiling_R2_on_proxy"]))
    cm = conv["measured"]
    print("  measured: model %.3f - causal GP %.3f = +%.3f ; MSE_h/MSE_pers(GP)=%.3f -> implied partial R2=%.4f (r=%.3f, I=%.4f nat)"
          % (cm["skill_model"], cm["skill_gp_causal"], cm["gain"], cm["mse_h_over_mse_pers_gp_causal"],
             cm["implied_R2_partial"], cm["implied_r_partial"], cm["implied_MI_nats"]))
    r_gp_tmp = conv["measured"]["mse_h_over_mse_pers_gp_causal"]
    pred = {}
    for nm in ("mod3", "ch15", "ch15+inc15", "ch15 lags0-5(75)"):
        for fit in ("robust", "classical", "gp_causal_measured"):
            ratio = conv["measured"]["mse_h_over_mse_pers_gp_causal"] if fit == "gp_causal_measured" else conv[fit]["mse_h_over_mse_pers"]
            pred[(nm, fit)] = {"in": multi[nm]["r2_in"] * ratio, "oos": multi[nm]["r2_oos_mean"] * ratio}
    print("\npredicted max linear dskill vs persistence (R2 x MSE_h/MSE_pers):")
    for (nm, fit), v in pred.items():
        print("  %-12s %-18s in-sample %.4f   oos %.4f" % (nm, fit, v["in"], v["oos"]))
    out["pred_dskill"] = {"%s|%s" % k: v for k, v in pred.items()}
    print("predicted max linear dskill from PARTIAL R2 (given prev level), GP ratio %.3f:" % r_gp_tmp)
    for nm, v in partial.items():
        print("  %-18s in %.4f  oos %.4f" % (nm, v["r2_partial_in"] * r_gp_tmp, v["r2_partial_oos_mean"] * r_gp_tmp))
        v["dskill_gp_ratio_oos"] = v["r2_partial_oos_mean"] * r_gp_tmp
    # tail-weighted decomposition for 30 regressors
    w30 = cond["ch15+inc15"]
    r_gp = conv["measured"]["mse_h_over_mse_pers_gp_causal"]
    dec = {"tail": sse_share_tail * w30["tail"]["explained_share_joint_fit"] * r_gp,
           "rest": (1 - sse_share_tail) * w30["rest"]["explained_share_joint_fit"] * r_gp}
    out["dskill_decomposition_joint_fit_gp_ratio"] = dec
    print("  joint-fit dskill decomposition (GP ratio): tail %.4f + rest %.4f = %.4f" % (dec["tail"], dec["rest"], dec["tail"] + dec["rest"]))

    # ---- single-modality rate-correlation bound (S8ar numbers) ----
    for nm, r in (("BES_rate_8ar", 0.070), ("ECEI_rate_8ar", 0.078), ("BES_level_8ar", 0.341), ("ECEI_level_8ar", 0.311)):
        out.setdefault("8ar_bounds", {})[nm] = {"r": r, "r2": r * r, "dskill_gp_ratio": r * r * r_gp}
        print("  8ar %-14s r=%.3f r2=%.4f -> dskill<=%.4f" % (nm, r, r * r, r * r * r_gp))

    with open(OUT_DIR / "p06_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("\nwrote", OUT_DIR / "p06_results.json")


if __name__ == "__main__":
    main()
