"""P10 -- real-time budget from plasma time scales.

Part 1: OU latency-cost table from the fitted T_i (and V_rot) parameters in
        data/.foundations.json (robust + classical fits).
Part 2: empirical impulse census from the raw CSVs -- top 1 % one-step CES_TI drops
        (held-free, 3 keV cut, same-block consecutive 10 ms pairs), amplitude in eV and
        as a fraction of the pre-drop value, recovery time in grid steps, event rate.
Part 3: operator / MAC count of one seq_v2 stateful step and the cost-model floor.

Read-only: nothing under the repo is written. ASCII-only output (cp949 console).
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(r"C:\Users\lss\Documents\GitHub\ml-intern-revision")
DATA = Path(os.environ.get("CES_DATA_DIR", REPO / "data"))
OUT_DIR = Path(__file__).resolve().parent
STUCK_GAP_SECONDS = 0.5
CUT_EV = 3000.0
DT = 0.010
OUT = {}


def hr(t):
    print("")
    print("=" * 78)
    print(t)
    print("=" * 78)


# ----------------------------------------------------------------------------
# Part 1 -- OU latency cost
# ----------------------------------------------------------------------------
def kalman_steady(s_n2, s_s2, rho):
    p = s_s2
    for _ in range(20000):
        p_minus = rho ** 2 * p + s_s2 * (1 - rho ** 2)
        kg = p_minus / (p_minus + s_n2)
        p_new = (1 - kg) * p_minus
        if abs(p_new - p) < 1e-14:
            p = p_new
            break
        p = p_new
    return p, kg


def part1():
    hr("PART 1 -- OU latency-cost table (skill vs latency ell)")
    fj = json.load(open(DATA / ".foundations.json"))
    ells_ms = [0.0, 1.0, 1.5, 2.0, 5.0, 6.0, 9.0, 10.0, 20.0, 50.0]
    res = {}
    for tgt, unit in (("CES_TI", "eV"), ("CES_VT", "km/s")):
        res[tgt] = {}
        for key in ("robust", "classical"):
            f = fj["D"][tgt]["fits"][key]
            s_n, s_s, tau = f["sigma_n"], f["sigma_s"], f["tau_ms"] / 1e3
            s_n2, s_s2 = s_n ** 2, s_s ** 2
            rho1 = np.exp(-DT / tau)
            P, K = kalman_steady(s_n2, s_s2, rho1)
            mse_pers = 2 * s_n2 + 2 * s_s2 * (1 - rho1)
            mse_interp = f["pred"]["mse_interp_opt"]
            # sanity: ell = 10 ms must reproduce the stored one-step filter MSE
            mse_10 = rho1 ** 2 * P + s_s2 * (1 - rho1 ** 2) + s_n2
            assert abs(mse_10 - f["pred"]["mse_filter_inf"]) < 1e-6 * mse_10, (mse_10, f["pred"]["mse_filter_inf"])
            rows = []
            print("")
            print("%s [%s] %s fit: sigma_n=%.2f sigma_s=%.2f tau=%.1f ms rho(10ms)=%.4f P=%.1f K=%.3f"
                  % (tgt, unit, key, s_n, s_s, tau * 1e3, rho1, P, K))
            print("  MSE_pers=%.1f (RMSE %.2f)  MSE_interp_opt=%.1f  (units %s^2)" % (mse_pers, np.sqrt(mse_pers), mse_interp, unit))
            if tgt == "CES_TI":
                sk_model = 0.358   # measured model skill vs persistence, s42 cut, dt<=15 ms (theory_foundations sec. 8 table)
                mse0 = P + s_n2; sk0_tmp = 1 - mse0 / mse_pers
                slope_tmp = (2.0 / tau) * (s_s2 - P) / 1e3 / mse_pers
                print("  equivalent age of the measured model skill (+0.358 vs pers): (%.4f - 0.358) / %.5f per ms = %.1f ms" % (sk0_tmp, slope_tmp, (sk0_tmp - sk_model) / slope_tmp))
            print("  %8s %10s %10s %8s %10s %10s %10s %8s" % ("ell[ms]", "rho^2(ell)", "MSE(ell)", "RMSE", "skill_pers", "skill_int", "d_skill", "retain"))
            for ell in ells_ms:
                r2 = np.exp(-2 * ell / 1e3 / tau)
                mse = r2 * P + s_s2 * (1 - r2) + s_n2
                sk_p = 1 - mse / mse_pers
                sk_i = 1 - mse / mse_interp
                rows.append({"ell_ms": ell, "rho2": float(r2), "mse": float(mse),
                             "skill_vs_pers": float(sk_p), "skill_vs_interp": float(sk_i)})
            sk0 = rows[0]["skill_vs_pers"]
            for r in rows:
                r["dskill_vs_ell0"] = r["skill_vs_pers"] - sk0
                r["rmse"] = float(np.sqrt(r["mse"])); r["retained"] = r["skill_vs_pers"] / sk0
                print("  %8.1f %10.4f %10.1f %8.2f %+10.4f %+10.4f %+10.4f %8.3f"
                      % (r["ell_ms"], r["rho2"], r["mse"], r["rmse"], r["skill_vs_pers"], r["skill_vs_interp"], r["dskill_vs_ell0"], r["retained"]))
            # small-ell linearisation: d MSE / d ell = (2/tau)(sigma_s^2 - P) at ell=0
            slope = (2.0 / tau) * (s_s2 - P)            # per second
            print("  linear slope d(MSE)/d(ell) at 0 = %.2f %s^2 per ms -> d(skill)/d(ell) = %.5f per ms"
                  % (slope / 1e3, unit, -slope / 1e3 / mse_pers))
            # latency at which skill vs persistence drops by 0.01 and 0.05
            def ell_for_loss(loss):
                lo, hi = 0.0, 5.0
                for _ in range(200):
                    mid = 0.5 * (lo + hi)
                    r2 = np.exp(-2 * mid / tau)
                    mse = r2 * P + s_s2 * (1 - r2) + s_n2
                    if (sk0 - (1 - mse / mse_pers)) < loss:
                        lo = mid
                    else:
                        hi = mid
                return 1e3 * mid
            e1, e5 = ell_for_loss(0.01), ell_for_loss(0.05)
            print("  latency costing 0.01 skill: %.1f ms ; 0.05 skill: %.1f ms" % (e1, e5))
            res[tgt][key] = {"sigma_n": s_n, "sigma_s": s_s, "tau_ms": tau * 1e3, "rho_10ms": float(rho1),
                             "P_filtered": float(P), "kalman_gain": float(K), "mse_pers": float(mse_pers),
                             "mse_interp_opt": float(mse_interp), "rows": rows,
                             "slope_mse_per_ms": float(slope / 1e3), "slope_skill_per_ms": float(-slope / 1e3 / mse_pers),
                             "ell_ms_for_0.01": float(e1), "ell_ms_for_0.05": float(e5)}
    OUT["part1"] = res


# ----------------------------------------------------------------------------
# Part 2 -- empirical impulse census
# ----------------------------------------------------------------------------
def _stuck_mask(values, block):
    mask = np.zeros(values.shape[0], dtype=bool)
    obs = np.flatnonzero(~np.isnan(values))
    if obs.size <= 1:
        return mask
    ov, ob = values[obs], block[obs]
    dup = (ov[1:] == ov[:-1]) & (ob[1:] == ob[:-1])
    mask[obs[1:][dup]] = True
    return mask


def load_ti():
    files = sorted(DATA.glob("*.csv"))
    if not files:
        raise SystemExit("no shot CSVs found")
    series = []
    for p in files:
        df = pd.read_csv(p, usecols=["time", "CES_TI"])
        t = df["time"].to_numpy(dtype=float)
        v = df["CES_TI"].to_numpy(dtype=float).copy()
        block = np.concatenate(([0], np.cumsum(np.diff(t) >= STUCK_GAP_SECONDS)))
        v[v > CUT_EV] = np.nan
        v[_stuck_mask(v, block)] = np.nan
        series.append((p.name, t, block, v))
    return series


def part2():
    hr("PART 2 -- top 1 % one-step CES_TI drops (held-free, 3 keV cut) and their recovery")
    series = load_ti()
    print("files: %d" % len(series))
    incs, pre, loc = [], [], []
    n_obs_total, n_pairs = 0, 0
    for fi, (name, t, block, v) in enumerate(series):
        n_obs_total += int(np.isfinite(v).sum())
        if v.size < 2:
            continue
        a, b = v[:-1], v[1:]
        same = block[:-1] == block[1:]
        step = np.abs((t[1:] - t[:-1]) - DT) < 0.5 * DT
        m = same & step & np.isfinite(a) & np.isfinite(b)
        n_pairs += int(m.sum())
        idx = np.flatnonzero(m)
        incs.append(b[idx] - a[idx])
        pre.append(a[idx])
        loc.append(np.stack([np.full(idx.size, fi), idx], 1))
    incs = np.concatenate(incs); pre = np.concatenate(pre); loc = np.concatenate(loc)
    neg = incs < 0
    print("consecutive same-block 10 ms pairs: %d ; negative increments: %d (%.1f%%)"
          % (n_pairs, neg.sum(), 100 * neg.mean()))
    q01 = np.quantile(incs[neg], 0.01)        # most negative 1 % of the negative increments
    sel = neg & (incs <= q01)
    drops = -incs[sel]; pre_sel = pre[sel]; frac = drops / pre_sel
    print("top-1%% threshold of negative increments: drop >= %.1f eV ; n_events = %d" % (-q01, sel.sum()))
    print("  drop amplitude [eV]: median %.1f  p90 %.1f  max %.1f" % (np.median(drops), np.quantile(drops, 0.9), drops.max()))
    print("  pre-drop level [eV]: median %.1f" % np.median(pre_sel))
    print("  drop / pre-drop:     median %.3f  p90 %.3f" % (np.median(frac), np.quantile(frac, 0.9)))
    # for context: the bulk one-step increment
    print("  bulk |increment| median %.1f eV, p99 of |increment| %.1f eV" % (np.median(np.abs(incs)), np.quantile(np.abs(incs), 0.99)))

    # recovery: steps until value returns (a) to >= pre - 0.2*drop  (80 % of the drop recovered)
    #                                     (b) to >= 0.8*pre          (literal 'within 20 % of pre-drop level')
    # plus a provenance check: was the pre-drop value itself a one-point excursion above its own
    # recent past (spike-descent, a fit failure the 3 keV cut does not catch)?
    CAP = 100
    rec_a, rec_b, next_nan, never_a, never_b, block_end = [], [], 0, 0, 0, 0
    cls = []            # 'spike_descent' | 'recovering' | 'step_down'
    pre_vs_past = []    # (pre - median of previous <=5 observed) / drop
    post_vs_past = []   # (post - median of previous <=5 observed) / drop
    ev_rows = []
    for (fi, i), d, p0 in zip(loc[sel].astype(int), drops, pre_sel):
        name, t, block, v = series[fi]
        j = i + 1                       # index of the post-drop value
        if j + 1 < v.size and np.isnan(v[j + 1]) and block[j + 1] == block[j]:
            next_nan += 1
        # recent past before the pre-drop row, same block, up to 5 observed values
        past = []
        for jj in range(i - 1, max(-1, i - 40), -1):
            if block[jj] != block[i]:
                break
            if np.isfinite(v[jj]):
                past.append(v[jj])
            if len(past) == 5:
                break
        ka = kb = None
        k = 0
        for jj in range(j + 1, min(v.size, j + 1 + CAP)):
            if block[jj] != block[j]:
                block_end += 1
                break
            k = int(round((t[jj] - t[j]) / DT))
            if np.isfinite(v[jj]):
                if ka is None and v[jj] >= p0 - 0.2 * d:
                    ka = k
                if kb is None and v[jj] >= 0.8 * p0:
                    kb = k
            if ka is not None and kb is not None:
                break
        if ka is None: never_a += 1
        else: rec_a.append(ka)
        if kb is None: never_b += 1
        else: rec_b.append(kb)
        if past:
            m_past = float(np.median(past))
            r_pre = (p0 - m_past) / d
            r_post = (v[j] - m_past) / d
        else:
            r_pre = r_post = np.nan
        pre_vs_past.append(r_pre); post_vs_past.append(r_post)
        if np.isfinite(r_pre) and r_pre > 0.5:
            c = "spike_descent"      # the pre-drop value sat > half a drop above its own past
        elif ka is not None and ka <= 50:
            c = "recovering"         # 80 % of the drop regained within 500 ms
        else:
            c = "step_down"          # no recovery within the cap / block
        cls.append(c)
        ev_rows.append({"file": name, "row": int(i), "time": float(t[i]), "pre": float(p0), "drop": float(d),
                        "rec_a": ka, "r_pre": r_pre, "class": c})
    rec_a = np.array(rec_a); rec_b = np.array(rec_b)
    n_ev = int(sel.sum())
    cls = np.array(cls); pre_vs_past = np.array(pre_vs_past); post_vs_past = np.array(post_vs_past)
    print("  provenance: (pre - median of <=5 previous obs)/drop: median %.2f ; share > 0.5 (pre-drop value was itself an excursion): %.1f%%"
          % (np.nanmedian(pre_vs_past), 100 * np.nanmean(pre_vs_past > 0.5)))
    print("              (post - median of <=5 previous obs)/drop: median %.2f (0 = the drop merely returns to the recent level)"
          % np.nanmedian(post_vs_past))
    class_stats = {}
    for c in ("spike_descent", "recovering", "step_down"):
        mk = cls == c
        if not mk.any():
            continue
        dd, ff = drops[mk], frac[mk]
        ra = np.array([r["rec_a"] for r in ev_rows if r["class"] == c and r["rec_a"] is not None])
        print("  class %-14s n=%4d (%.1f%%)  drop median %.0f eV p90 %.0f eV ; drop/pre median %.2f p90 %.2f ; rec(a) median %s steps (n=%d)"
              % (c, mk.sum(), 100 * mk.mean(), np.median(dd), np.quantile(dd, 0.9), np.median(ff), np.quantile(ff, 0.9),
                 ("%.0f" % np.median(ra)) if ra.size else "-", ra.size))
        class_stats[c] = {"n": int(mk.sum()), "drop_median": float(np.median(dd)), "drop_p90": float(np.quantile(dd, 0.9)),
                          "frac_median": float(np.median(ff)), "frac_p90": float(np.quantile(ff, 0.9)),
                          "rec_a_median": float(np.median(ra)) if ra.size else None, "rec_a_p90": float(np.quantile(ra, 0.9)) if ra.size else None,
                          "mean_A2": float(np.mean(dd ** 2))}
    OUT["part2_classes"] = class_stats
    # physically-sized ELM-like census: the recovering class only
    mk = cls == "recovering"
    if mk.any():
        ra = np.array([r["rec_a"] for r in ev_rows if r["class"] == "recovering"])
        print("  ELM-like (recovering) census: n=%d, rate %.3f /s, drop median %.0f eV (%.0f%% of level), recovery median %.0f steps -> tau_rec %.0f ms"
              % (mk.sum(), mk.sum() / (n_obs_total * DT), np.median(drops[mk]), 100 * np.median(frac[mk]), np.median(ra), 10 * np.median(ra) / np.log(5.0)))
        A2r = float(np.mean(drops[mk] ** 2)); rr = mk.sum() / (n_obs_total * DT)
        for ell in (1.5, 9.0):
            print("    continuous-use latency charge r*ell*A^2 at ell=%.1f ms: %.0f eV^2 (%.1f%% of robust MSE_pers)" % (ell, rr * ell / 1e3 * A2r, 100 * rr * ell / 1e3 * A2r / 4136.55))
        OUT["part2_elm_like"] = {"n": int(mk.sum()), "rate_per_s": float(rr), "drop_median": float(np.median(drops[mk])),
                                 "frac_median": float(np.median(frac[mk])), "rec_median_steps": float(np.median(ra)),
                                 "rec_p90_steps": float(np.quantile(ra, 0.9)), "mean_A2": A2r}
    print("  recovery (a) 80%% of the drop regained: median %.0f steps (%.0f ms), p90 %.0f steps; recovered %d/%d, not within %d steps or block end %d"
          % (np.median(rec_a), 10 * np.median(rec_a), np.quantile(rec_a, 0.9), rec_a.size, n_ev, CAP, never_a))
    print("  recovery (b) value >= 0.8 * pre-drop:   median %.0f steps, p90 %.0f steps; recovered %d/%d; events whose drop < 20%% of level (b trivially met at k=1): %d"
          % (np.median(rec_b), np.quantile(rec_b, 0.9), rec_b.size, n_ev, int((frac < 0.2).sum())))
    print("  next grid row after the drop is missing (fit rejected / MNAR candidate): %d/%d (%.1f%%)"
          % (next_nan, n_ev, 100 * next_nan / n_ev))
    # event rate per second of observed series (observed rows * 10 ms)
    obs_seconds = n_obs_total * DT
    rate = n_ev / obs_seconds
    print("  observed T_i rows: %d (%.0f s) -> impulse rate %.3f /s (one per %.2f s)" % (n_obs_total, obs_seconds, rate, 1 / rate))
    # MSE charged by a latency ell to a step of amplitude A at rate r: r * ell * A^2 (time-averaged)
    A2 = float(np.mean(drops ** 2))
    print("  mean A^2 over events = %.0f eV^2 ; latency MSE charge r*ell*A^2:" % A2)
    for ell in (1.0, 1.5, 5.0, 9.0, 10.0):
        print("    ell=%4.1f ms: %.2f eV^2  (robust bulk MSE_pers 4136.6 -> %.3f%% ; classical 44598 -> %.4f%%)"
              % (ell, rate * ell / 1e3 * A2, 100 * rate * ell / 1e3 * A2 / 4136.55, 100 * rate * ell / 1e3 * A2 / 44598.3))
    # recovery relative error of a delayed sensor: exp recovery, tau_rec = median recovery / ln(5) (80 % at median)
    tau_rec_ms = 10 * np.median(rec_a) / np.log(5.0)
    print("  exponential recovery time constant from median (a): tau_rec = %.1f ms" % tau_rec_ms)
    for ell in (1.0, 1.5, 5.0, 9.0, 10.0):
        print("    delayed-sensor peak relative error during recovery ~ 1-exp(-ell/tau_rec) = %.3f at ell=%.1f ms" % (1 - np.exp(-ell / tau_rec_ms), ell))
    # secondary census sized to the ELM physics (theory_plasma_physics 6.3: 5-20 % of pedestal energy):
    # drops of 10-50 % of the pre-drop level, >= 150 eV, pre-drop row not an excursion, 80 % regained within 50 steps
    OUT["part2_moderate"] = {}
    for MIN_EV in (150.0, 420.0):
        mod = neg & (-incs >= 0.10 * pre) & (-incs <= 0.50 * pre) & (-incs >= MIN_EV)
        n_mod, n_mod_rec, mod_drops, mod_frac, mod_rec = 0, 0, [], [], []
        for (fi, i), inc, p0 in zip(loc[mod].astype(int), incs[mod], pre[mod]):
            name, t, block, v = series[fi]
            d = -inc; j = i + 1
            past = []
            for jj in range(i - 1, max(-1, i - 40), -1):
                if block[jj] != block[i]:
                    break
                if np.isfinite(v[jj]):
                    past.append(v[jj])
                if len(past) == 5:
                    break
            if not past or (p0 - float(np.median(past))) / d > 0.5:
                continue
            n_mod += 1
            ka = None
            for jj in range(j + 1, min(v.size, j + 51)):
                if block[jj] != block[j]:
                    break
                if np.isfinite(v[jj]) and v[jj] >= p0 - 0.2 * d:
                    ka = int(round((t[jj] - t[j]) / DT)); break
            if ka is not None:
                n_mod_rec += 1; mod_drops.append(d); mod_frac.append(d / p0); mod_rec.append(ka)
        mod_drops = np.array(mod_drops); mod_frac = np.array(mod_frac); mod_rec = np.array(mod_rec)
        print("  moderate ELM-sized census (10-50%% of level, >=%.0f eV, no pre-excursion): candidates %d, recovering within 50 steps %d (rate %.3f /s)"
              % (MIN_EV, n_mod, n_mod_rec, n_mod_rec / (n_obs_total * DT)))
        print("    drop median %.0f eV p90 %.0f eV ; drop/pre median %.2f p90 %.2f ; recovery median %.0f steps p90 %.0f steps -> tau_rec %.0f ms"
              % (np.median(mod_drops), np.quantile(mod_drops, 0.9), np.median(mod_frac), np.quantile(mod_frac, 0.9),
                 np.median(mod_rec), np.quantile(mod_rec, 0.9), 10 * np.median(mod_rec) / np.log(5.0)))
        A2m = float(np.mean(mod_drops ** 2)); rm = n_mod_rec / (n_obs_total * DT)
        for ell in (1.5, 9.0):
            print("    continuous-use latency charge at ell=%.1f ms: %.0f eV^2 (%.1f%% of robust MSE_pers)" % (ell, rm * ell / 1e3 * A2m, 100 * rm * ell / 1e3 * A2m / 4136.55))
        OUT["part2_moderate"][str(int(MIN_EV))] = {"candidates": n_mod, "recovering": n_mod_rec, "rate_per_s": float(rm),
                                 "drop_median": float(np.median(mod_drops)), "drop_p90": float(np.quantile(mod_drops, 0.9)),
                                 "frac_median": float(np.median(mod_frac)), "frac_p90": float(np.quantile(mod_frac, 0.9)),
                                 "rec_median_steps": float(np.median(mod_rec)), "rec_p90_steps": float(np.quantile(mod_rec, 0.9)), "mean_A2": A2m}
    OUT["part2"] = {"n_files": len(series), "n_pairs": n_pairs, "n_negative": int(neg.sum()),
                    "threshold_drop_ev": float(-q01), "n_events": n_ev,
                    "drop_ev": {"median": float(np.median(drops)), "p90": float(np.quantile(drops, 0.9)), "max": float(drops.max())},
                    "pre_level_ev_median": float(np.median(pre_sel)),
                    "drop_frac": {"median": float(np.median(frac)), "p90": float(np.quantile(frac, 0.9))},
                    "bulk_abs_inc_median": float(np.median(np.abs(incs))), "bulk_abs_inc_p99": float(np.quantile(np.abs(incs), 0.99)),
                    "recovery_a_steps": {"median": float(np.median(rec_a)), "p90": float(np.quantile(rec_a, 0.9)), "n": int(rec_a.size), "never": int(never_a)},
                    "recovery_b_steps": {"median": float(np.median(rec_b)), "p90": float(np.quantile(rec_b, 0.9)), "n": int(rec_b.size), "never": int(never_b),
                                         "trivial_drop_lt_20pct": int((frac < 0.2).sum())},
                    "next_row_missing": next_nan, "obs_rows": n_obs_total, "impulse_rate_per_s": float(rate),
                    "mean_A2": A2, "tau_rec_ms": float(tau_rec_ms)}


# ----------------------------------------------------------------------------
# Part 3 -- one stateful seq_v2 step: MACs, minimal dispatch count, cost-model floor
# ----------------------------------------------------------------------------
def part3():
    hr("PART 3 -- one seq_v2 stateful step: MACs, dispatches, cost floor")
    n_in, n_fast = 22, 15
    h1, L1, h2, L2, head = 160, 2, 64, 1, 64
    n_slow = n_in - n_fast
    mac_ti = 4 * h1 * (n_in + h1) + (L1 - 1) * 4 * h1 * (h1 + h1)
    mac_vt = 4 * h2 * (n_slow + h2)
    mac_heads = (h1 * head + head * 1) + (h2 * head + head * 1)
    mac = mac_ti + mac_vt + mac_heads
    print("LSTM_ti: in %d hidden %d layers %d -> %d MAC ; LSTM_vt: in %d hidden %d -> %d MAC ; heads %d MAC ; total %d MAC"
          % (n_in, h1, L1, mac_ti, n_slow, h2, mac_vt, mac_heads, mac))
    for gf in (5.0, 20.0):
        print("  arithmetic at %.0f GFLOP/s (2 FLOP/MAC): %.3f ms" % (gf, 2 * mac / (gf * 1e9) * 1e3))
    # minimal dispatch count of the fused step (TightSeqV2Step structure)
    per_layer = {"cat(x,h)": 1, "addmm": 1, "slice sigmoid part": 1, "sigmoid": 1, "slice tanh part": 1, "tanh": 1,
                 "slice f": 1, "mul f*c": 1, "slice i": 1, "mul i*g": 1, "add": 1, "slice o": 1, "tanh(c)": 1, "mul o*tanh(c)": 1}
    n_layer_ops = sum(per_layer.values())
    layers = 2   # fused layer-0 for both branches + deep layer-1 of the T_i branch
    tail = {"split h -> (ti, vt)": 2, "layer_norm x2": 2, "head addmm x2 per branch": 4, "gelu x2": 2, "cat output": 1}
    n_tail = sum(tail.values())
    n_min = layers * n_layer_ops + n_tail
    print("  per-layer dispatches in a fused LSTM step: %d ; layers executed per step: %d ; tail: %d ; minimal count: %d"
          % (n_layer_ops, layers, n_tail, n_min))
    oc = json.load(open(DATA / ".b9_op_counts.json"))
    mn = json.load(open(DATA / ".b9_latency_min.json"))
    for arm in ("seq_v2", "seq_v2_lean", "seq_v2_tight", "persistence", "gp_causal", "window_w2"):
        ops = oc.get(arm, {}).get("aten_ops")
        m = mn[arm]
        print("  %-13s aten_ops=%4s  min_median=%.3f ms  min_p99=%.3f ms  max_p99=%.3f ms  spread=%.1fx"
              % (arm, ops, m["min_median_ms"], m["min_p99_ms"], m["max_p99_ms"], m["spread"]))
    for arm in ("seq_v2_tight", "seq_v2_lean"):
        ops = oc[arm]["aten_ops"]
        print("  cost-model floor for %s: %d ops x 2-3 us = %.2f-%.2f ms ; measured min median %.3f ms (%.2f us/op)"
              % (arm, ops, ops * 2e-3, ops * 3e-3, mn[arm]["min_median_ms"], 1e3 * mn[arm]["min_median_ms"] / ops))
    print("  minimal-count floor: %d ops x 2-3 us = %.2f-%.2f ms" % (n_min, n_min * 2e-3, n_min * 3e-3))
    lb = json.load(open(DATA / ".latency_benchmark.json"))
    for r in lb["runs"]:
        if r["device"] == "cpu" and r["batch"] == 1:
            print("  8l artifact (re-measured 2026-08-17): %-16s W=%s median %.3f ms p99 %.3f ms (%.1f%% of 10 ms)"
                  % (r["model"], r["window"], r["median_ms"], r["p99_ms"], r["grid_budget_used_pct_p99"]))
    OUT["part3"] = {"mac_total": mac, "mac_ti": mac_ti, "mac_vt": mac_vt, "mac_heads": mac_heads,
                    "min_dispatch_count": n_min, "per_layer": n_layer_ops, "tail": n_tail,
                    "op_counts": {a: oc[a]["aten_ops"] for a in ("seq_v2", "seq_v2_lean", "seq_v2_tight")},
                    "latency_min": {a: mn[a] for a in ("seq_v2", "seq_v2_lean", "seq_v2_tight", "persistence", "gp_causal", "window_w2")}}


# ----------------------------------------------------------------------------
# Part 4 -- explicit time-domain forms: skill(ell, g) and the ELM crash/recovery error e(t)
# ----------------------------------------------------------------------------
def part4():
    hr("PART 4 -- time-domain forms: skill(ell, g) and the delayed-sensor error through an ELM")
    fj = json.load(open(DATA / ".foundations.json"))
    res = {"skill_ell_g": {}, "elm": {}}
    for key in ("robust", "classical"):
        f = fj["D"]["CES_TI"]["fits"][key]
        s_n2, s_s2, tau = f["sigma_n"] ** 2, f["sigma_s"] ** 2, f["tau_ms"] / 1e3
        rho1 = np.exp(-DT / tau)
        P, _ = kalman_steady(s_n2, s_s2, rho1)
        print("")
        print("CES_TI %s fit: skill(ell, g) = 1 - MSE_now(ell) / MSE_pers(g); nowcaster depends on ell only, persistence on g only" % key)
        print("  %8s" % "ell|g[ms]" + "".join("%10d" % g for g in (10, 20, 50, 100)))
        tab = {}
        for ell in (0.0, 1.5, 6.0, 9.0):
            row = []
            for g in (10, 20, 50, 100):
                r2l = np.exp(-2 * ell / 1e3 / tau); rg = np.exp(-g / 1e3 / tau)
                mse_now = r2l * P + s_s2 * (1 - r2l) + s_n2
                mse_pers_g = 2 * s_n2 + 2 * s_s2 * (1 - rg)
                row.append(1 - mse_now / mse_pers_g)
            tab[ell] = row
            print("  %8.1f" % ell + "".join("%+10.3f" % v for v in row))
        # CES-only filter whose last observation is g old: MSE_filter(g) = rho_g^2 P + s_s2(1-rho_g^2) + s_n2 (foundations 7.7 by_gap)
        print("  CES-only causal filter, last observation g old (latency irrelevant once ell <= g):")
        for g in (10, 20, 50):
            r2g = np.exp(-2 * g / 1e3 / tau)
            mse_f = r2g * P + s_s2 * (1 - r2g) + s_n2
            stored = f["pred"]["by_gap"][str(g // 10)]["mse_causal"]
            print("    g=%3d ms: MSE %.1f (stored by_gap %.1f)" % (g, mse_f, stored))
        res["skill_ell_g"][key] = {str(k): v for k, v in tab.items()}

    # ELM trajectory and delayed-sensor error (deterministic, onset unknown before t0, sensor replays truth delayed by ell)
    def traj(t, y0, A, tau_c, t_c, tau_r, t0=0.0):
        y = np.full_like(t, y0)
        t1 = t0 + t_c
        ymin = y0 - A * (1 - np.exp(-t_c / tau_c))
        m1 = (t > t0) & (t <= t1); m2 = t > t1
        y[m1] = y0 - A * (1 - np.exp(-(t[m1] - t0) / tau_c))
        y[m2] = ymin + (y0 - ymin) * (1 - np.exp(-(t[m2] - t1) / tau_r))
        return y, ymin
    dt = 1e-6
    t = np.arange(-0.002, 0.400, dt)
    y0 = 1234.0   # pre-drop level median of the >=420 eV census (not used in error, only the shape)
    for (A, tau_r_ms, label) in ((509.0, 25.0, "3sigma census: A=509 eV, tau_rec=25 ms"), (1198.0, 37.0, "recovering top-1%: A=1198 eV, tau_rec=37 ms")):
        for tau_c_ms, t_c_ms in ((0.3, 1.0),):
            tau_c, t_c, tau_r = tau_c_ms / 1e3, t_c_ms / 1e3, tau_r_ms / 1e3
            y, ymin = traj(t, y0, A, tau_c, t_c, tau_r)
            E_event = float(np.sum((y - y0) ** 2) * dt)    # eV^2 s : energy of the excursion itself
            print("")
            print("ELM: %s ; tau_crash=%.1f ms, t_c=%.1f ms -> y_min = y0 - %.1f eV ; excursion energy int (y-y0)^2 dt = %.1f eV^2 s" % (label, tau_c_ms, t_c_ms, y0 - ymin, E_event))
            print("  %6s %12s %12s %14s %14s %12s" % ("ell", "peak|e| eV", "peak at ms", "int e^2 dt", "frac of event", "e at t1+ell"))
            out = {}
            for ell_ms in (0.0, 1.0, 1.5, 6.0, 9.0, 10.0):
                ell = ell_ms / 1e3
                yd, _ = traj(t - ell, y0, A, tau_c, t_c, tau_r)   # y(t - ell)
                e = y - yd
                ipk = int(np.argmax(np.abs(e)))
                ise = float(np.sum(e ** 2) * dt)
                e_rec = (y0 - ymin) * (1 - np.exp(-ell / tau_r))  # closed form at t = t1 + ell
                print("  %6.1f %12.1f %12.2f %14.2f %14.3f %12.1f" % (ell_ms, abs(e[ipk]), 1e3 * t[ipk], ise, ise / E_event, e_rec))
                out[str(ell_ms)] = {"peak_abs_e": float(abs(e[ipk])), "peak_t_ms": float(1e3 * t[ipk]), "ise": ise, "frac": ise / E_event, "e_rec_closed": float(e_rec)}
            res["elm"][label] = {"tau_c_ms": tau_c_ms, "t_c_ms": t_c_ms, "ymin_drop": float(y0 - ymin), "E_event": E_event, "rows": out}
            # 10 ms label integration: the first post-onset grid row sees only a phase-dependent fraction of A
            fr = []
            for phase in np.linspace(0, DT, 101)[:-1]:
                mk = (t > phase - DT) & (t <= phase)
                fr.append((y0 - y[mk].mean()) / (y0 - ymin))
            print("  10 ms label integration: first grid row after onset shows (y0 - ybar)/(y0 - ymin) = %.2f (min) .. %.2f (max), mean %.2f over onset phase" % (min(fr), max(fr), np.mean(fr)))
            res["elm"][label]["label_fraction_first_row"] = {"min": float(min(fr)), "max": float(max(fr)), "mean": float(np.mean(fr))}
    OUT["part4"] = res


if __name__ == "__main__":
    part1()
    part2()
    part3()
    part4()
    out = OUT_DIR / "p10_results.json"
    json.dump(OUT, open(out, "w"), indent=1)
    print("")
    print("wrote %s" % out)
