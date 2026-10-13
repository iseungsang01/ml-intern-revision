"""P08 -- per-discharge win-rate asymmetry from the two generative models.

Reads only the frozen TEST artifacts of the four canonical B.1 seq_v2 runs and the raw
shot CSVs.  Trains nothing, scores nothing.  All prints are ASCII.

Outputs: p08_results.json next to this script.
"""
import glob
import json
import os
import sys

import numpy as np

REPO = r"C:\Users\lss\Documents\GitHub\ml-intern-revision"
DATA = os.path.join(REPO, "data")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
SEEDS = ["42", "1", "7", "123"]
TARGETS = {"CES_TI": 1, "CES_VT": 2}
MIN_ROWS = 20          # same rule as shot_covariates.py (sec. 8an)
TOP_FRAC = 0.05        # "impulse" rows = top 5 % |dy| within the discharge
Z_THR = 1.0            # fast-channel visibility threshold on |d z|
RNG = np.random.default_rng(20261013)

FILES = sorted(glob.glob(os.path.join(DATA, "s*.csv")))
assert len(FILES) == 641, len(FILES)
HEADER = open(FILES[0], encoding="utf-8").readline().strip().split(",")
BES_COLS = [i for i, h in enumerate(HEADER) if h.startswith("BES_")]
ECEI_COLS = [i for i, h in enumerate(HEADER) if h.startswith("ECEI_")]
MC_COLS = [i for i, h in enumerate(HEADER) if h.startswith("MC")]

_csv_cache = {}


def load_csv(idx):
    if idx not in _csv_cache:
        arr = np.genfromtxt(FILES[idx], delimiter=",", skip_header=1)
        _csv_cache[idx] = arr
    return _csv_cache[idx]


def match_rows(csv_col, y):
    """Map npz y_true (float32 round-trip) to CSV rows, monotone in row order.

    Returns array of row indices (-1 where unmatched)."""
    rows = np.full(len(y), -1, dtype=np.int64)
    last = -1
    cand_all = np.where(np.isfinite(csv_col))[0]
    for i, v in enumerate(y):
        tol = 1e-3 * abs(v) + 0.05
        ok = cand_all[(np.abs(csv_col[cand_all] - v) <= tol) & (cand_all > last)]
        if len(ok) == 0:
            continue
        rows[i] = ok[0]
        last = ok[0]
    return rows


def per_shot_z(fast):
    mean = np.nanmean(fast, axis=0)
    std = np.nanstd(fast, axis=0)
    std[std < 1e-6] = 1.0
    return (fast - mean) / std


def discharge_table(target, col):
    """One row per (split, shot) pair, plus the raw per-row arrays for pooled checks."""
    recs = []
    per_row = {"split": [], "shot": [], "sm": [], "sg": [], "sp": [], "peak": []}
    for seed in SEEDS:
        z = np.load(os.path.join(DATA, f".b1_seqv2_s{seed}_i{seed}", "comparison_errors_test.npz"))
        sh = z[f"{target}_shot"]
        y = z[f"{target}_y_true"].astype(np.float64)
        sm = z[f"{target}_se_model"].astype(np.float64)
        sg = z[f"{target}_se_gp_causal"].astype(np.float64)
        sp = z[f"{target}_se_pchip"].astype(np.float64)
        pk = z[f"{target}_is_peak"].astype(bool)
        dt = z[f"{target}_dt_ms"].astype(np.float64)
        ok = np.isfinite(sm) & np.isfinite(sg) & np.isfinite(sp)
        for s in np.unique(sh):
            m = (sh == s) & ok
            n = int(m.sum())
            if n < MIN_ROWS or sg[m].sum() <= 0:
                continue
            ys, sms, sgs, sps, pks, dts = y[m], sm[m], sg[m], sp[m], pk[m], dt[m]
            # impulse share: fraction of sum(dy^2) carried by the top-5 % |dy| rows
            dy = np.diff(ys)
            k = max(1, int(np.ceil(TOP_FRAC * len(dy))))
            order = np.argsort(-np.abs(dy))
            top = order[:k]
            ss_all = float(np.sum(dy ** 2))
            imp_share = float(np.sum(dy[top] ** 2) / ss_all) if ss_all > 0 else np.nan
            # same rows, for the SSE decomposition: "impulse rows" = target row i+1 of the top dy
            imp_rows = np.zeros(len(ys), dtype=bool)
            imp_rows[top + 1] = True
            # ---- CSV-based visibility (p_obs proxy) ----
            arr = load_csv(int(s))
            rows = match_rows(arr[:, col], ys)
            matched = rows >= 0
            fast_z = per_shot_z(arr[:, BES_COLS + ECEI_COLS])
            n_bes = len(BES_COLS)
            vis_e = np.full(len(ys), np.nan)
            vis_b = np.full(len(ys), np.nan)
            vis_mc = np.full(len(ys), np.nan)
            mc_z = per_shot_z(arr[:, MC_COLS])
            for i in range(1, len(ys)):
                if rows[i] < 1 or rows[i - 1] < 0:
                    continue
                # change of the fast channels across the SAME pair of CES rows as dy
                d = fast_z[rows[i]] - fast_z[rows[i - 1]]
                vis_b[i] = np.nanmax(np.abs(d[:n_bes]))
                vis_e[i] = np.nanmax(np.abs(d[n_bes:]))
                dm = mc_z[rows[i]] - mc_z[rows[i - 1]]
                vis_mc[i] = np.nanmax(np.abs(dm))
            imp_idx = top + 1
            v_e = vis_e[imp_idx]
            v_b = vis_b[imp_idx]
            v_all = vis_e[1:]
            p_obs_e = float(np.nanmean(v_e > Z_THR)) if np.isfinite(v_e).any() else np.nan
            p_obs_b = float(np.nanmean(v_b > Z_THR)) if np.isfinite(v_b).any() else np.nan
            p_obs_eb = float(np.nanmean((v_e > Z_THR) | (v_b > Z_THR))) if np.isfinite(v_e).any() else np.nan
            base_e = float(np.nanmean(v_all > Z_THR)) if np.isfinite(v_all).any() else np.nan
            base_eb = float(np.nanmean((vis_e[1:] > Z_THR) | (vis_b[1:] > Z_THR))) if np.isfinite(v_all).any() else np.nan
            # continuous version: rank correlation |dy| vs |dECEI_z| inside the discharge
            fin = np.isfinite(vis_e[1:])
            if fin.sum() > 10:
                from_rank = lambda a: np.argsort(np.argsort(a))
                a, b = np.abs(dy[fin]), vis_e[1:][fin]
                rho_dy_e = float(np.corrcoef(from_rank(a), from_rank(b))[0, 1])
            else:
                rho_dy_e = np.nan
            recs.append({
                "split": seed, "shot_idx": int(s),
                "shot": int(os.path.basename(FILES[int(s)])[1:].split(".")[0]),
                "n": n,
                "win_gp": bool(sms.sum() < sgs.sum()),
                "win_pchip": bool(sms.sum() < sps.sum()),
                "skill_gp": float(1 - sms.sum() / sgs.sum()),
                "D_gp": float(sgs.mean() - sms.mean()),          # MSE_base - MSE_model
                "D_pchip": float(sps.mean() - sms.mean()),
                "spread": float(np.std(ys)),
                "var": float(np.var(ys)),
                "imp_share": imp_share,
                "peak_frac": float(pks.mean()),
                # peak-row / non-peak-row wins
                "win_gp_peak": bool(sms[pks].sum() < sgs[pks].sum()) if pks.any() else None,
                "win_gp_nonpeak": bool(sms[~pks].sum() < sgs[~pks].sum()) if (~pks).any() else None,
                "sse_share_peak_model": float(sms[pks].sum() / sms.sum()) if sms.sum() > 0 else np.nan,
                "sse_share_peak_gp": float(sgs[pks].sum() / sgs.sum()),
                # impulse-row wins (top-5 % |dy| rows)
                "win_gp_imp": bool(sms[imp_rows].sum() < sgs[imp_rows].sum()),
                "win_gp_nonimp": bool(sms[~imp_rows].sum() < sgs[~imp_rows].sum()),
                "D_gp_imp_contrib": float((sgs[imp_rows].sum() - sms[imp_rows].sum()) / n),
                "D_gp_nonimp_contrib": float((sgs[~imp_rows].sum() - sms[~imp_rows].sum()) / n),
                "matched_frac": float(matched.mean()),
                "p_obs_ecei": p_obs_e, "p_obs_bes": p_obs_b, "p_obs_any": p_obs_eb,
                "base_ecei": base_e, "base_any": base_eb,
                "rho_dy_decei": rho_dy_e,
                "mean_dt_ms": float(dts.mean()),
            })
    return recs


def tercile_rates(recs, key, win="win_gp"):
    x = np.array([r[key] for r in recs], dtype=float)
    w = np.array([r[win] for r in recs], dtype=float)
    ok = np.isfinite(x)
    x, w = x[ok], w[ok]
    q = np.quantile(x, [1 / 3, 2 / 3])
    out = []
    for lo, hi, name in ((-np.inf, q[0], "quiet"), (q[0], q[1], "middle"), (q[1], np.inf, "variable")):
        m = (x > lo) & (x <= hi)
        out.append({"tercile": name, "n": int(m.sum()), "win_rate": float(w[m].mean()),
                    "x_range": [float(x[m].min()), float(x[m].max())]})
    return out


def perm_corr(x, y, n_perm=10000):
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    r = float(np.corrcoef(x, y)[0, 1])
    cnt = 0
    for _ in range(n_perm):
        if abs(np.corrcoef(RNG.permutation(x), y)[0, 1]) >= abs(r):
            cnt += 1
    return r, (cnt + 1) / (n_perm + 1), int(ok.sum())


def spearman(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    rx = np.argsort(np.argsort(x[ok])); ry = np.argsort(np.argsort(y[ok]))
    return float(np.corrcoef(rx, ry)[0, 1])


def logistic_fit(X, y, iters=200, l2=1e-4):
    """Plain Newton logistic regression; X without intercept (added here)."""
    X = np.column_stack([np.ones(len(y)), X])
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ b))
        g = X.T @ (p - y) + l2 * b
        H = (X * (p * (1 - p))[:, None]).T @ X + l2 * np.eye(len(b))
        step = np.linalg.solve(H, g)
        b -= step
        if np.max(np.abs(step)) < 1e-9:
            break
    p = 1 / (1 + np.exp(-X @ b))
    se = np.sqrt(np.diag(np.linalg.inv((X * (p * (1 - p))[:, None]).T @ X + l2 * np.eye(len(b)))))
    return b, se


def theil_sen(x, y):
    """Robust slope/intercept (median of pairwise slopes)."""
    n = len(x)
    i, j = np.triu_indices(n, 1)
    dx = x[j] - x[i]
    ok = dx != 0
    slope = float(np.median((y[j] - y[i])[ok] / dx[ok]))
    inter = float(np.median(y - slope * x))
    return slope, inter


def main():
    results = {}
    for target, col in TARGETS.items():
        recs = discharge_table(target, col)
        n = len(recs)
        wins = np.array([r["win_gp"] for r in recs], float)
        winp = np.array([r["win_pchip"] for r in recs], float)
        spread = np.array([r["spread"] for r in recs])
        var = np.array([r["var"] for r in recs])
        imp = np.array([r["imp_share"] for r in recs])
        D = np.array([r["D_gp"] for r in recs])
        skill = np.array([r["skill_gp"] for r in recs])
        phys = sorted({r["shot"] for r in recs})
        print(f"\n===== {target} =====")
        print(f"(split,shot) discharges: {n}   physical shots: {len(phys)}   "
              f"double-counted pairs: {n - len(phys)}")
        print(f"win rate vs gp_causal: {wins.mean():.3f}   vs pchip: {winp.mean():.3f}   "
              f"median skill vs gp: {np.median(skill):+.3f}")
        # physical-shot pooled table (sec. 8an convention) for the record
        by_phys = {}
        for r in recs:
            by_phys.setdefault(r["shot"], []).append(r)
        phys_win = np.mean([np.mean([rr["win_gp"] for rr in v]) for v in by_phys.values()])
        print(f"physical-shot averaged win rate vs gp: {phys_win:.3f}")

        # (a) terciles of spread
        ter_spread = tercile_rates(recs, "spread")
        ter_spread_p = tercile_rates(recs, "spread", "win_pchip")
        print("win rate vs gp by tercile of target spread:")
        for t, tp in zip(ter_spread, ter_spread_p):
            print(f"  {t['tercile']:8s} n={t['n']:3d}  win_gp={t['win_rate']:.3f}  "
                  f"win_pchip={tp['win_rate']:.3f}  spread=[{t['x_range'][0]:.1f},{t['x_range'][1]:.1f}]")
        # (b) impulse share
        ter_imp = tercile_rates(recs, "imp_share")
        print("win rate vs gp by tercile of impulse share (top-5% |dy| share of sum dy^2):")
        for t in ter_imp:
            print(f"  {t['tercile']:8s} n={t['n']:3d}  win_gp={t['win_rate']:.3f}  "
                  f"share=[{t['x_range'][0]:.3f},{t['x_range'][1]:.3f}]")
        r_sw, p_sw, _ = perm_corr(spread, wins)
        r_iw, p_iw, _ = perm_corr(imp, wins)
        print(f"corr(spread, win) = {r_sw:+.3f} (perm p={p_sw:.4f});  "
              f"corr(imp_share, win) = {r_iw:+.3f} (perm p={p_iw:.4f})")
        # logistic: win ~ log spread + imp_share
        Xl = np.column_stack([np.log(spread), imp])
        okl = np.isfinite(Xl).all(axis=1)
        b, se = logistic_fit(Xl[okl], wins[okl])
        print(f"logit P(win) = {b[0]:+.2f} ({se[0]:.2f}) + {b[1]:+.2f} ({se[1]:.2f}) * ln(spread) "
              f"+ {b[2]:+.2f} ({se[2]:.2f}) * imp_share")
        # ceiling: win rate in the top decile of spread
        top10 = spread >= np.quantile(spread, 0.9)
        top20 = spread >= np.quantile(spread, 0.8)
        print(f"ceiling: win rate in top-10% spread = {wins[top10].mean():.3f} (n={top10.sum()}), "
              f"top-20% = {wins[top20].mean():.3f} (n={top20.sum()})")
        # D vs var: sign of intercept and slope (prediction: a<0, b>0)
        slope, inter = theil_sen(var, D)
        rho_vD = spearman(var, D)
        print(f"D = MSE_gp - MSE_model vs var(y): Theil-Sen intercept {inter:+.1f}, slope {slope:+.4f}; "
              f"spearman rho = {rho_vD:+.3f}")
        print(f"  median D in quiet tercile {np.median(D[spread <= np.quantile(spread, 1/3)]):+.1f}, "
              f"in variable tercile {np.median(D[spread > np.quantile(spread, 2/3)]):+.1f}")
        # impulse-row vs bulk contribution to D
        Dimp = np.array([r["D_gp_imp_contrib"] for r in recs])
        Dbulk = np.array([r["D_gp_nonimp_contrib"] for r in recs])
        print(f"D decomposition (mean over discharges): impulse rows {Dimp.mean():+.1f}, "
              f"bulk rows {Dbulk.mean():+.1f};  medians {np.median(Dimp):+.1f} / {np.median(Dbulk):+.1f}")
        print(f"  share of discharges with bulk contribution < 0: {(Dbulk < 0).mean():.3f};  "
              f"impulse contribution > 0: {(Dimp > 0).mean():.3f}")
        win_imp = np.array([r["win_gp_imp"] for r in recs], float)
        win_nonimp = np.array([r["win_gp_nonimp"] for r in recs], float)
        print(f"  win on impulse rows {win_imp.mean():.3f}, win on bulk rows {win_nonimp.mean():.3f}")
        # (c) peak rows
        wp = np.array([r["win_gp_peak"] for r in recs if r["win_gp_peak"] is not None], float)
        wn = np.array([r["win_gp_nonpeak"] for r in recs if r["win_gp_nonpeak"] is not None], float)
        sp_m = np.array([r["sse_share_peak_model"] for r in recs])
        sp_g = np.array([r["sse_share_peak_gp"] for r in recs])
        pf = np.array([r["peak_frac"] for r in recs])
        print(f"peak rows: mean peak fraction {pf.mean():.3f};  win on peak rows {wp.mean():.3f} (n={len(wp)}), "
              f"win on non-peak rows {wn.mean():.3f} (n={len(wn)})")
        ratio = sp_m / sp_g
        okr = np.isfinite(ratio)
        print(f"  peak-row SSE share: model {np.nanmean(sp_m):.3f}, gp {np.nanmean(sp_g):.3f}, "
              f"ratio model/gp median {np.median(ratio[okr]):.3f}")
        # p_obs proxy from CSVs
        mf = np.array([r["matched_frac"] for r in recs])
        pe = np.array([r["p_obs_ecei"] for r in recs])
        pb = np.array([r["p_obs_bes"] for r in recs])
        pa = np.array([r["p_obs_any"] for r in recs])
        be = np.array([r["base_ecei"] for r in recs])
        ba = np.array([r["base_any"] for r in recs])
        rde = np.array([r["rho_dy_decei"] for r in recs])
        print(f"CSV row matching: mean matched fraction {mf.mean():.3f} (min {mf.min():.3f})")
        print(f"p_obs proxy (top-5% |dy| rows with |dECEI_z|>1): mean {np.nanmean(pe):.3f}, median {np.nanmedian(pe):.3f}; "
              f"base rate over all rows {np.nanmean(be):.3f};  enrichment {np.nanmean(pe)/np.nanmean(be):.2f}")
        print(f"  BES: {np.nanmean(pb):.3f};  ECEI or BES: {np.nanmean(pa):.3f} vs base {np.nanmean(ba):.3f} "
              f"(enrichment {np.nanmean(pa)/np.nanmean(ba):.2f})")
        print(f"  within-discharge spearman(|dy|, |dECEI_z|): mean {np.nanmean(rde):+.3f}, "
              f"median {np.nanmedian(rde):+.3f}, share > 0: {(rde > 0).mean():.3f}")
        # does p_obs predict the win beyond spread?  (within the variable tercile)
        var_t = spread > np.quantile(spread, 2 / 3)
        hi = pa > np.nanmedian(pa[var_t])
        print(f"  variable tercile: win rate when p_obs_any above its median {wins[var_t & hi].mean():.3f} "
              f"(n={(var_t & hi).sum()}) vs below {wins[var_t & ~hi].mean():.3f} (n={(var_t & ~hi).sum()})")
        r_pw, p_pw, _ = perm_corr(pa[var_t], wins[var_t])
        print(f"  corr(p_obs_any, win | variable tercile) = {r_pw:+.3f} (perm p={p_pw:.4f})")
        # pooled-enrichment the other way: among rows with visible fast change, what is |dy|?
        results[target] = {
            "n_pairs": n, "n_phys": len(phys), "win_gp": float(wins.mean()), "win_pchip": float(winp.mean()),
            "phys_win_gp": float(phys_win), "median_skill_gp": float(np.median(skill)),
            "tercile_spread_gp": ter_spread, "tercile_spread_pchip": ter_spread_p,
            "tercile_imp": ter_imp,
            "corr_spread_win": [r_sw, p_sw], "corr_imp_win": [r_iw, p_iw],
            "logit": {"coef": b.tolist(), "se": se.tolist()},
            "ceiling_top10": float(wins[top10].mean()), "ceiling_top20": float(wins[top20].mean()),
            "theil_sen_D_vs_var": {"intercept": inter, "slope": slope, "spearman": rho_vD},
            "D_imp_mean": float(Dimp.mean()), "D_bulk_mean": float(Dbulk.mean()),
            "D_imp_median": float(np.median(Dimp)), "D_bulk_median": float(np.median(Dbulk)),
            "bulk_neg_share": float((Dbulk < 0).mean()), "imp_pos_share": float((Dimp > 0).mean()),
            "win_imp_rows": float(win_imp.mean()), "win_bulk_rows": float(win_nonimp.mean()),
            "peak": {"frac": float(pf.mean()), "win_peak": float(wp.mean()), "win_nonpeak": float(wn.mean()),
                     "sse_share_model": float(np.nanmean(sp_m)), "sse_share_gp": float(np.nanmean(sp_g)),
                     "ratio_median": float(np.median(ratio[okr]))},
            "p_obs": {"ecei": float(np.nanmean(pe)), "ecei_median": float(np.nanmedian(pe)),
                      "bes": float(np.nanmean(pb)), "any": float(np.nanmean(pa)),
                      "base_ecei": float(np.nanmean(be)), "base_any": float(np.nanmean(ba)),
                      "rho_dy_decei_mean": float(np.nanmean(rde)), "rho_pos_share": float((rde > 0).mean()),
                      "var_tercile_hi": float(wins[var_t & hi].mean()), "var_tercile_lo": float(wins[var_t & ~hi].mean()),
                      "corr_pobs_win_var_tercile": [r_pw, p_pw], "matched_frac": float(mf.mean())},
            "discharges": recs,
        }
    with open(os.path.join(OUT_DIR, "p08_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1, default=lambda o: None if (isinstance(o, float) and np.isnan(o)) else o)
    print("\nwrote p08_results.json")


if __name__ == "__main__":
    main()
