"""P02 -- V_rot verdict as the consequence of the toroidal momentum balance with
every torque term unobserved.  Read-only: frozen TEST npz + raw CSVs + OU fit.
No model is trained or scored.  ASCII-only output (cp949 console)."""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = r"C:\Users\lss\Documents\GitHub\ml-intern-revision"
NPZ = os.path.join(ROOT, "data", ".b1_seqv2_s42_i42", "comparison_errors_test.npz")
FOUND = os.path.join(ROOT, "data", ".foundations.json")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "p02_results.json")

res = {}


# ---------------------------------------------------------------- Part A: npz
def sse_shares(se_m, se_p, q):
    """Top-q rows ordered by se_m + se_p.  Returns shares and ratios."""
    order = np.argsort(-(se_m + se_p))
    k = max(1, int(np.ceil(q * len(order))))
    top = np.zeros(len(order), bool)
    top[order[:k]] = True
    Sm, Sp = se_m.sum(), se_p.sum()
    s_p = se_p[top].sum() / Sp               # PCHIP tail share
    s_m = se_m[top].sum() / Sm               # model tail share
    r_T = se_m[top].sum() / se_p[top].sum()  # tail SSE ratio model/PCHIP
    r_B = se_m[~top].sum() / se_p[~top].sum()  # bulk SSE ratio model/PCHIP
    skill_recon = 1.0 - (s_p * r_T + (1 - s_p) * r_B)
    lhs = s_p * (1.0 - r_T)            # tail gain (normalised by PCHIP SSE)
    rhs = (1.0 - s_p) * (r_B - 1.0)    # bulk deficit (normalised by PCHIP SSE)
    return dict(q=q, rows=int(k), share_pchip=float(s_p), share_model=float(s_m),
                r_tail=float(r_T), r_bulk=float(r_B), skill_recon=float(skill_recon),
                tail_gain=float(lhs), bulk_deficit=float(rhs), holds=bool(lhs > rhs),
                median_ratio_bulk=float(np.median(np.sqrt(se_m[~top])) / np.median(np.sqrt(se_p[~top]))))


def part_a():
    d = np.load(NPZ)
    out = {}
    for tgt in ("CES_VT", "CES_TI"):
        se = {k: d[f"{tgt}_se_{k}"].astype(float) for k in
              ("model", "pchip", "linear", "gp", "gp_causal", "persistence")}
        dt = d[f"{tgt}_dt_ms"]
        peak = d[f"{tgt}_is_peak"]
        n = len(dt)
        ae = {k: np.sqrt(v) for k, v in se.items()}
        med = {k: float(np.median(v)) for k, v in ae.items()}
        mse = {k: float(v.mean()) for k, v in se.items()}
        o = dict(n=n, median_ae=med, mse=mse,
                 median_ratio_model_over_pchip=med["model"] / med["pchip"],
                 median_ratio_model_over_pers=med["model"] / med["persistence"],
                 median_ratio_gpc_over_pers=med["gp_causal"] / med["persistence"],
                 median_ratio_pchip_over_pers=med["pchip"] / med["persistence"],
                 skill_vs_pchip=1 - mse["model"] / mse["pchip"],
                 skill_vs_pers=1 - mse["model"] / mse["persistence"],
                 skill_vs_gpc=1 - mse["model"] / mse["gp_causal"],
                 skill_gpc_vs_pchip=1 - mse["gp_causal"] / mse["pchip"],
                 win_rate_vs_pchip=float((se["model"] < se["pchip"]).mean()),
                 win_rate_vs_pers=float((se["model"] < se["persistence"]).mean()))
        # one-step rows (dt <= 15 ms), the ladder of foundations E
        m1 = dt <= 15
        o["one_step"] = dict(
            n=int(m1.sum()),
            median_ratio_vs_pers={k: float(np.median(ae[k][m1]) / np.median(ae["persistence"][m1]))
                                  for k in ae},
            skill_vs_pers={k: float(1 - se[k][m1].mean() / se["persistence"][m1].mean()) for k in se},
            median_ratio_model_over_pchip=float(np.median(ae["model"][m1]) / np.median(ae["pchip"][m1])),
            skill_model_vs_pchip=float(1 - se["model"][m1].mean() / se["pchip"][m1].mean()))
        # tail shares & inequality
        o["tail"] = {str(q): sse_shares(se["model"], se["pchip"], q) for q in (0.01, 0.05, 0.10, 0.25)}
        # running contribution curve (foundations 10.3 identity)
        order = np.argsort(se["model"] + se["pchip"])
        cm, cp = np.cumsum(se["model"][order]), np.cumsum(se["pchip"][order])
        Sp = se["pchip"].sum()
        o["running_contribution"] = {str(p): float((cp[int(p * n) - 1] - cm[int(p * n) - 1]) / Sp)
                                     for p in (0.5, 0.75, 0.9, 0.99, 1.0)}
        o["quantile_ratio_model_over_pchip"] = {
            str(p): float(np.quantile(ae["model"], p) / np.quantile(ae["pchip"], p))
            for p in (0.5, 0.75, 0.9, 0.95, 0.99)}
        # which tail rows: PCHIP hurt by its future anchor (se_pchip > se_pers) vs genuine jump
        order2 = np.argsort(-(se["model"] + se["pchip"]))
        k1 = int(np.ceil(0.01 * n))
        top = order2[:k1]
        anchor_bad = se["pchip"][top] > se["persistence"][top]
        o["tail_1pct_anatomy"] = dict(
            rows=k1,
            frac_rows_pchip_worse_than_pers=float(anchor_bad.mean()),
            frac_pchip_sse_in_those_rows=float(se["pchip"][top][anchor_bad].sum() / se["pchip"][top].sum()),
            median_abs_y_true_jump_proxy_pers_ae=float(np.median(ae["persistence"][top])),
            median_pers_ae_all=med["persistence"],
            model_vs_pers_skill_in_tail=float(1 - se["model"][top].sum() / se["persistence"][top].sum()),
            model_vs_pchip_skill_in_tail=float(1 - se["model"][top].sum() / se["pchip"][top].sum()))
        # peak stratum
        o["peak"] = {}
        for name, m in (("peak", peak), ("nonpeak", ~peak)):
            o["peak"][name] = dict(n=int(m.sum()),
                                   skill_vs_pchip=float(1 - se["model"][m].mean() / se["pchip"][m].mean()),
                                   median_ratio=float(np.median(ae["model"][m]) / np.median(ae["pchip"][m])))
        # gap bins
        o["by_gap"] = {}
        for lo, hi in ((0, 15), (15, 25), (25, 45), (45, 1e9)):
            m = (dt > lo) & (dt <= hi)
            if m.sum() < 5:
                continue
            o["by_gap"][f"({lo},{hi}]"] = dict(
                n=int(m.sum()),
                skill_model_vs_pchip=float(1 - se["model"][m].mean() / se["pchip"][m].mean()),
                skill_model_vs_pers=float(1 - se["model"][m].mean() / se["persistence"][m].mean()),
                median_ratio_model_over_pchip=float(np.median(ae["model"][m]) / np.median(ae["pchip"][m])),
                median_ratio_model_over_pers=float(np.median(ae["model"][m]) / np.median(ae["persistence"][m])),
                median_ratio_gpc_over_pchip=float(np.median(ae["gp_causal"][m]) / np.median(ae["pchip"][m])))
        out[tgt] = o
    return out


# ---------------------------------------------------------- Part B: OU theory
def part_b():
    f = json.load(open(FOUND, encoding="utf-8"))
    out = {}
    for tgt in ("CES_VT", "CES_TI"):
        fit = f["D"][tgt]["fits"]["robust"]
        sn, ss, tau = fit["sigma_n"], fit["sigma_s"], fit["tau_ms"]
        dt = 10.0
        rho = np.exp(-dt / tau)
        # steady-state Kalman posterior variance P after an observation (10 ms spacing)
        P = ss ** 2
        for _ in range(10000):
            Pm = rho ** 2 * P + ss ** 2 * (1 - rho ** 2)
            K = Pm / (Pm + sn ** 2)
            P = (1 - K) * Pm
        var_pers_1 = 2 * sn ** 2 + 2 * ss ** 2 * (1 - rho)

        def var_causal(g):
            r = np.exp(-g / tau)
            return r ** 2 * P + ss ** 2 * (1 - r ** 2) + sn ** 2

        def var_pers(g):
            r = np.exp(-g / tau)
            return 2 * sn ** 2 + 2 * ss ** 2 * (1 - r)

        def var_bridge(gp, gf):
            ra, rb = np.exp(-gp / tau), np.exp(-gf / tau)
            c = ss ** 2 * np.array([ra, rb])
            S = ss ** 2 * np.array([[1, ra * rb], [ra * rb, 1]]) + sn ** 2 * np.eye(2)
            return ss ** 2 + sn ** 2 - c @ np.linalg.solve(S, c)

        def var_bridge_noisefree(gp, gf):
            ra, rb = np.exp(-gp / tau), np.exp(-gf / tau)
            return ss ** 2 * (1 - ra ** 2) * (1 - rb ** 2) / (1 - ra ** 2 * rb ** 2)

        gaps = [10, 20, 30, 50, 80, 100, 200]
        tab = {}
        for g in gaps:
            vc, vp, vb = var_causal(g), var_pers(g), var_bridge(g, 10)
            tab[str(g)] = dict(var_causal=float(vc), var_pers=float(vp), var_bridge_g_10=float(vb),
                               var_bridge_g_g=float(var_bridge(g, g)),
                               bridge_noisefree_g_g=float(var_bridge_noisefree(g, g)),
                               skill_causal_vs_pers=float(1 - vc / vp),
                               skill_causal_vs_bridge=float(1 - vc / vb),
                               rmse_ratio_causal_over_pers=float(np.sqrt(vc / vp)))
        out[tgt] = dict(sigma_n=sn, sigma_s=ss, tau_ms=tau, rho_10ms=float(rho), P_post=float(P),
                        kalman_gain=float(K), var_filter_1step=float(var_causal(dt)),
                        var_pers_1step=float(var_pers_1),
                        rmse_ratio_filter_over_pers=float(np.sqrt(var_causal(dt) / var_pers_1)),
                        rmse_ratio_bridge_over_pers=float(np.sqrt(var_bridge(dt, dt) / var_pers_1)),
                        dynamics_only_skill=float((1 - rho) / 2),
                        by_gap=tab)
    return out


# ------------------------------------------------------ Part C: raw CSV rates
GAP_S = 0.5
MIN_PAIRS = 40
LAGS = (-2, -1, 0, 1, 2)


def stuck_mask(values, block):
    stuck = np.zeros(values.shape, bool)
    last_val, last_block = np.nan, -1
    for i, v in enumerate(values):
        if np.isnan(v):
            continue
        if block[i] == last_block and not np.isnan(last_val) and v == last_val:
            stuck[i] = True
        else:
            last_val, last_block = v, block[i]
    return stuck


def modality_mean(df, cols):
    a = df[cols].to_numpy(float)
    mu = np.nanmean(a, axis=0, keepdims=True)
    sd = np.nanstd(a, axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    return np.nanmean((a - mu) / sd, axis=1)


def increment(v):
    d = np.full(v.shape, np.nan)
    ok = ~np.isnan(v)
    pair = ok[1:] & ok[:-1]
    d[1:][pair] = v[1:][pair] - v[:-1][pair]
    return d


class Pool:
    def __init__(self):
        self.sxy = self.sxx = self.syy = 0.0
        self.n = 0
        self.blocks = 0
        self.block_r = []

    def add(self, x, y):
        x = x - x.mean()
        y = y - y.mean()
        self.sxy += (x * y).sum()
        self.sxx += (x * x).sum()
        self.syy += (y * y).sum()
        self.n += len(x)
        self.blocks += 1
        den = np.sqrt((x * x).sum() * (y * y).sum())
        if den > 0:
            self.block_r.append(float((x * y).sum() / den))

    def r(self):
        den = np.sqrt(self.sxx * self.syy)
        return float(self.sxy / den) if den > 0 else float("nan")

    def summary(self):
        br = np.array(self.block_r)
        return dict(r=self.r(), n_pairs=int(self.n), n_blocks=int(self.blocks),
                    se_naive=float(1 / np.sqrt(max(self.n, 1))),
                    block_median_r=float(np.median(br)) if len(br) else float("nan"),
                    frac_blocks_positive=float((br > 0).mean()) if len(br) else float("nan"),
                    # block-clustered standard error of the pooled r via per-block mean
                    se_block=float(br.std(ddof=1) / np.sqrt(len(br))) if len(br) > 1 else float("nan"))


def part_c():
    files = sorted(glob.glob(os.path.join(ROOT, "data", "*.csv")))
    pools = {}          # (tgt, drv, lag, kind) -> Pool
    held_stats = dict(CES_VT=dict(obs=0, held=0), CES_TI=dict(obs=0, held=0))
    n_blocks_total = 0
    for p in files:
        df = pd.read_csv(p)
        if not {"time", "CES_TI", "CES_VT"} <= set(df.columns):
            continue
        t = df["time"].to_numpy(float)
        block = np.concatenate(([0], np.cumsum(np.diff(t) >= GAP_S)))
        groups = {m: [c for c in df.columns if c.startswith(m)] for m in ("BES", "ECEI", "MC")}
        drv = {m: modality_mean(df, cols) for m, cols in groups.items() if cols}
        tg = {}
        for c in ("CES_TI", "CES_VT"):
            v = df[c].to_numpy(float).copy()
            if c == "CES_TI":
                v[v > 3000] = np.nan
            obs = np.isfinite(v)
            sm = stuck_mask(v, block)
            held_stats[c]["obs"] += int(obs.sum())
            held_stats[c]["held"] += int(sm.sum())
            v[sm] = np.nan
            tg[c] = v
        dtg = {c: increment(v) for c, v in tg.items()}
        ddr = {m: increment(v) for m, v in drv.items()}
        for b in np.unique(block):
            sel = np.where(block == b)[0]
            if len(sel) < MIN_PAIRS:
                continue
            n_blocks_total += 1
            for c in ("CES_TI", "CES_VT"):
                y = dtg[c][sel]
                ylev = tg[c][sel]
                for m in ddr:
                    x = ddr[m][sel]
                    xlev = drv[m][sel]
                    for lag in LAGS:
                        # pair dV_t with dX_{t+lag}
                        if lag >= 0:
                            yy, xx = y[:len(y) - lag] if lag else y, x[lag:]
                        else:
                            yy, xx = y[-lag:], x[:len(x) + lag]
                        ok = np.isfinite(yy) & np.isfinite(xx)
                        if ok.sum() < MIN_PAIRS:
                            continue
                        pools.setdefault((c, m, lag, "rate"), Pool()).add(xx[ok], yy[ok])
                        pools.setdefault((c, m, lag, "abs_rate"), Pool()).add(np.abs(xx[ok]), np.abs(yy[ok]))
                    ok = np.isfinite(ylev) & np.isfinite(xlev)
                    if ok.sum() >= MIN_PAIRS:
                        pools.setdefault((c, m, 0, "level"), Pool()).add(xlev[ok], ylev[ok])
    out = dict(n_files=len(files), n_blocks_ge40=n_blocks_total, held=held_stats, corr={})
    for (c, m, lag, kind), pl in sorted(pools.items()):
        out["corr"][f"{c}|{m}|lag{lag:+d}|{kind}"] = pl.summary()
    return out


def main():
    res["A_npz"] = part_a()
    res["B_ou"] = part_b()
    res["C_raw"] = part_c()
    json.dump(res, open(OUT, "w"), indent=1)
    A, B, C = res["A_npz"], res["B_ou"], res["C_raw"]
    print("=== PART A: frozen TEST npz s42 cut ===")
    for tgt in ("CES_VT", "CES_TI"):
        o = A[tgt]
        print(f"[{tgt}] n={o['n']}  median |e| model/PCHIP={o['median_ratio_model_over_pchip']:.3f}"
              f"  model/pers={o['median_ratio_model_over_pers']:.3f}  gpc/pers={o['median_ratio_gpc_over_pers']:.3f}"
              f"  pchip/pers={o['median_ratio_pchip_over_pers']:.3f}")
        print(f"   MSE skill vs PCHIP={o['skill_vs_pchip']:+.4f} vs pers={o['skill_vs_pers']:+.4f}"
              f" vs gpc={o['skill_vs_gpc']:+.4f}  gpc vs PCHIP={o['skill_gpc_vs_pchip']:+.4f}"
              f"  winrate vs PCHIP={o['win_rate_vs_pchip']:.3f} vs pers={o['win_rate_vs_pers']:.3f}")
        os_ = o["one_step"]
        print(f"   one-step (dt<=15, n={os_['n']}): median/pers " +
              "  ".join(f"{k}={v:.3f}" for k, v in os_["median_ratio_vs_pers"].items()))
        print(f"   one-step skill vs pers: " + "  ".join(f"{k}={v:+.3f}" for k, v in os_["skill_vs_pers"].items()))
        for q, tl in o["tail"].items():
            print(f"   top {float(q)*100:4.1f}% ({tl['rows']} rows): share_pchip={tl['share_pchip']:.3f}"
                  f" share_model={tl['share_model']:.3f} r_tail={tl['r_tail']:.3f} r_bulk={tl['r_bulk']:.3f}"
                  f" skill_recon={tl['skill_recon']:+.4f} tail_gain={tl['tail_gain']:+.4f}"
                  f" bulk_deficit={tl['bulk_deficit']:+.4f} holds={tl['holds']}"
                  f" median_ratio_bulk={tl['median_ratio_bulk']:.3f}")
        print("   running contribution: " + "  ".join(f"q{k}={v:+.4f}" for k, v in o["running_contribution"].items()))
        print("   quantile ratio m/p: " + "  ".join(f"q{k}={v:.3f}" for k, v in o["quantile_ratio_model_over_pchip"].items()))
        print("   tail1% anatomy: " + json.dumps({k: round(v, 4) if isinstance(v, float) else v
                                                  for k, v in o["tail_1pct_anatomy"].items()}))
        print("   peak: " + json.dumps(o["peak"]))
        for g, v in o["by_gap"].items():
            print(f"   gap {g}: n={v['n']} skill m/p={v['skill_model_vs_pchip']:+.3f} m/pers={v['skill_model_vs_pers']:+.3f}"
                  f" med m/p={v['median_ratio_model_over_pchip']:.3f} med m/pers={v['median_ratio_model_over_pers']:.3f}"
                  f" med gpc/p={v['median_ratio_gpc_over_pchip']:.3f}")
    print("=== PART B: noisy-OU closed forms (robust fit, foundations D) ===")
    for tgt, o in B.items():
        print(f"[{tgt}] sn={o['sigma_n']:.2f} ss={o['sigma_s']:.2f} tau={o['tau_ms']:.1f} rho={o['rho_10ms']:.4f}"
              f" P={o['P_post']:.3f} K={o['kalman_gain']:.3f} filt/pers RMSE={o['rmse_ratio_filter_over_pers']:.3f}"
              f" bridge/pers RMSE={o['rmse_ratio_bridge_over_pers']:.3f} dyn-only skill={o['dynamics_only_skill']:.3f}")
        for g, v in o["by_gap"].items():
            print(f"   g={g:>3} ms: Var_causal={v['var_causal']:.2f} Var_pers={v['var_pers']:.2f}"
                  f" Var_bridge(g,10)={v['var_bridge_g_10']:.2f} Var_bridge(g,g)={v['var_bridge_g_g']:.2f}"
                  f" (noise-free {v['bridge_noisefree_g_g']:.2f}) skill c/pers={v['skill_causal_vs_pers']:+.3f}"
                  f" c/bridge={v['skill_causal_vs_bridge']:+.3f}")
    print("=== PART C: raw CSV, held-free increments, pooled Pearson (blocks >= 40 pairs) ===")
    print(f"files={C['n_files']} blocks={C['n_blocks_ge40']} held={C['held']}")
    for key, v in C["corr"].items():
        print(f"  {key:34s} r={v['r']:+.4f} se_block={v['se_block']:.4f} n={v['n_pairs']:7d} blocks={v['n_blocks']:4d}"
              f" blockmed={v['block_median_r']:+.4f} fracpos={v['frac_blocks_positive']:.3f}")
    print("json ->", OUT)


if __name__ == "__main__":
    main()
