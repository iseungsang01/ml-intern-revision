"""P03 -- 50 ms saturation: Kalman cold start (A), pedestal physics (B), ELM-like recovery on data (C).
Run from repo root:  py -I <this file>   (prints ASCII only; writes p03_results.json next to itself)
No model is trained or scored."""
import json, math, os
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get("CES_REPO", r"C:\Users\lss\Documents\GitHub\ml-intern-revision"))
DATA = REPO / "data"
DT = 0.010
STUCK_GAP_SECONDS = 0.5
CUT_EV = 3000.0
EPS = 0.02   # PRACTICAL_EPS (PREREGISTRATION_B9 section 3)
E_CH = 1.602176634e-19
OUT = {}

# measured ladder, THESIS_RESULTS section 8al table 1 (deficit vs full-block backbone, CES_TI)
MEAS = {2: -0.066, 3: -0.033, 4: -0.021, 5: -0.017, 6: -0.005, 7: +0.002, 10: -0.009, 15: +0.005}


# ---------------------------------------------------------------- A. cold-start Kalman
def kalman_transient(s_n, s_s, tau_ms, kmax=30):
    rho = math.exp(-DT / (tau_ms / 1e3))
    q = s_s ** 2 * (1 - rho ** 2)
    sn2 = s_n ** 2
    Pm = s_s ** 2
    for _ in range(10000):
        K = Pm / (Pm + sn2)
        P = (1 - K) * Pm
        Pm_new = rho ** 2 * P + q
        if abs(Pm_new - Pm) < 1e-12:
            break
        Pm = Pm_new
    Pm_inf = Pm
    K_inf = Pm / (Pm + sn2)
    mse_inf = Pm_inf + sn2
    ratio = (rho * (1 - K_inf)) ** 2
    L = 1.0 / (-math.log(rho * (1 - K_inf)))
    rows = []
    Pm = s_s ** 2                        # no-information prior before observation 1
    mse0 = Pm + sn2                      # k = 0 observations: predict the mean
    for k in range(1, kmax + 1):
        Kk = Pm / (Pm + sn2)
        P = (1 - Kk) * Pm
        Pm = rho ** 2 * P + q            # P^-_{k+1}
        rows.append({"k": k, "K_k": Kk, "Pminus_next": Pm, "mse_k": Pm + sn2,
                     "delta_k": Pm + sn2 - mse_inf,
                     "contraction_exact": rho ** 2 * (1 - Kk) * (1 - K_inf)})
    return dict(rho=rho, q=q, sn2=sn2, ss2=s_s ** 2, Pm_inf=Pm_inf, K_inf=K_inf, mse_inf=mse_inf,
                ratio=ratio, memory_steps=L, mse0=mse0, delta0=mse0 - mse_inf, rows=rows)


def steps_closed_form(delta0, ratio, eps_abs):
    """smallest real k with delta0 * ratio^k <= eps_abs (asymptotic-rate bound)"""
    return math.log(delta0 / eps_abs) / (-math.log(ratio))


def part_a():
    fj = json.load(open(DATA / ".foundations.json", encoding="utf-8"))
    fits = fj["D"]["CES_TI"]["fits"]
    A = {}
    for name in ("robust", "classical"):
        f = fits[name]
        sn, ss, tau = f["sigma_n"], f["sigma_s"], f["tau_ms"]
        kt = kalman_transient(sn, ss, tau)
        pred = f["pred"]
        mse_pchip = pred["mse_interp_opt"]
        mse_pers = pred["mse_persistence"]
        chk = {str(k): (kt["rows"][k - 1]["mse_k"], pred["mse_filter_K"][str(k)]) for k in (1, 2, 3, 5, 10)}
        refs = {"pchip_opt": mse_pchip, "filter_inf": kt["mse_inf"], "persistence": mse_pers}
        curves = {}
        for rn, ref in refs.items():
            curves[rn] = {k: kt["rows"][k - 1]["delta_k"] / ref for k in (1, 2, 3, 4, 5, 6, 7, 10, 15)}
        sat = {rn: min(k for k in range(1, 31) if kt["rows"][k - 1]["delta_k"] / ref < EPS) for rn, ref in refs.items()}
        kcf = {rn: steps_closed_form(kt["delta0"], kt["ratio"], EPS * ref) for rn, ref in refs.items()}
        rungs = [2, 3, 4, 5, 6, 7]
        comp = {}
        for rn in refs:
            pr = np.array([-curves[rn][k] for k in rungs])
            me = np.array([MEAS[k] for k in rungs])
            comp[rn] = {"rms_diff": float(np.sqrt(np.mean((pr - me) ** 2))),
                        "rms_meas": float(np.sqrt(np.mean(me ** 2))),
                        "rms_diff_k3to7": float(np.sqrt(np.mean((pr[1:] - me[1:]) ** 2)))}
        A[name] = dict(sigma_n=sn, sigma_s=ss, tau_ms=tau, rho=kt["rho"], K_inf=kt["K_inf"],
                       memory_steps=kt["memory_steps"], ratio_per_step=kt["ratio"],
                       mse_inf=kt["mse_inf"], mse0=kt["mse0"], delta0=kt["delta0"], refs=refs,
                       check_vs_json=chk, deficit_curves=curves, saturation_rung=sat,
                       closed_form_steps=kcf, shape_vs_8al=comp,
                       transient=[{"k": r["k"], "K_k": r["K_k"], "mse_k": r["mse_k"], "delta_k": r["delta_k"],
                                   "contraction_exact": r["contraction_exact"]} for r in kt["rows"][:10]])
    # geometric fit of the measured curve on k = 2..6 (all negative rungs)
    ks = np.array([2, 3, 4, 5, 6])
    ys = np.log(np.array([-MEAS[k] for k in ks]))
    slope, icpt = np.polyfit(ks, ys, 1)
    r_meas = math.exp(slope)
    L_meas = 1.0 / (-0.5 * math.log(r_meas))        # ratio = exp(-2/L)
    rho = A["robust"]["rho"]
    one_minus_K = math.sqrt(r_meas) / rho
    K_req = 1 - one_minus_K
    ss2 = 1.0
    Pm = ss2 * (1 - rho ** 2) / (1 - rho ** 2 * (1 - K_req))
    sn2_over_ss2 = Pm * (1 - K_req) / K_req / ss2
    # with that gain, how many steps to 2 % (of the filter's own steady state)?
    delta0_rel = (ss2 + sn2_over_ss2) - (Pm + sn2_over_ss2)      # mse0 - mse_inf in units of ss2
    k_req = steps_closed_form(delta0_rel, r_meas, EPS * (Pm + sn2_over_ss2))
    A["measured_curve_fit"] = dict(ratio_per_step=r_meas, amplitude=math.exp(icpt), memory_steps=L_meas,
                                   K_required_at_robust_rho=K_req,
                                   sigma_n_over_sigma_s_required=math.sqrt(sn2_over_ss2),
                                   steps_to_eps_with_that_gain=k_req,
                                   fit_pred={int(k): -math.exp(icpt + slope * k) for k in (2, 3, 4, 5, 6, 7, 10)})
    OUT["A"] = A
    print("A. cold-start Kalman (CES_TI)")
    for name in ("robust", "classical"):
        a = A[name]
        print("  %-9s sn=%.1f ss=%.1f tau=%.0f ms rho=%.3f K=%.3f memory=%.2f steps ratio=%.3f mse_inf=%.0f mse0=%.0f delta0=%.0f"
              % (name, a["sigma_n"], a["sigma_s"], a["tau_ms"], a["rho"], a["K_inf"], a["memory_steps"],
                 a["ratio_per_step"], a["mse_inf"], a["mse0"], a["delta0"]))
        print("    check vs json mse_filter_K:", {k: (round(v[0], 2), round(v[1], 2)) for k, v in a["check_vs_json"].items()})
        print("    transient K_k:", [round(r["K_k"], 3) for r in a["transient"][:6]],
              " exact contraction:", [round(r["contraction_exact"], 3) for r in a["transient"][:6]])
        for rn in a["refs"]:
            print("    ref=%-11s (%.0f) sat_rung=%d closed_form=%.2f rms_diff_vs_8al=%.4f (k3-7 %.4f) deficits:"
                  % (rn, a["refs"][rn], a["saturation_rung"][rn], a["closed_form_steps"][rn],
                     a["shape_vs_8al"][rn]["rms_diff"], a["shape_vs_8al"][rn]["rms_diff_k3to7"]),
                  {k: round(-v, 4) for k, v in a["deficit_curves"][rn].items()})
    m = A["measured_curve_fit"]
    print("  measured 8al curve: ratio/step=%.3f memory=%.2f steps amp=%.3f; needs K=%.3f (sn/ss=%.2f) at robust rho; steps to eps=%.2f"
          % (m["ratio_per_step"], m["memory_steps"], m["amplitude"], m["K_required_at_robust_rho"],
             m["sigma_n_over_sigma_s_required"], m["steps_to_eps_with_that_gain"]))
    print("    geometric fit of measured:", {k: round(v, 4) for k, v in m["fit_pred"].items()})


# ---------------------------------------------------------------- B. pedestal physics
def tau_eq_ms(te_ev, ne_m3, lnL=17.0, A_i=2.0):
    tau_e = 3.44e5 * te_ev ** 1.5 / (ne_m3 * 1e-6 * lnL)         # s  (NRL practical form)
    return tau_e * (A_i * 1836.15 / 2.0) * 1e3


def part_b():
    R, a, kappa, dped = 1.8, 0.5, 1.8, 0.03
    V_shell = 2 * math.pi * R * 2 * math.pi * a * math.sqrt(kappa) * dped
    V_plasma = 2 * math.pi ** 2 * R * a ** 2 * kappa
    P_loss = 2.0e6
    B = {"V_shell_m3": V_shell, "V_plasma_m3": V_plasma, "P_loss_MW": 2.0, "cases": []}
    for ne, te, ti in ((2e19, 500, 593), (3e19, 600, 593), (3e19, 1000, 1000), (5e19, 1000, 1000)):
        w_dens = 1.5 * ne * (te + ti) * E_CH
        B["cases"].append(dict(ne=ne, te=te, ti=ti, tau_eq_ms=tau_eq_ms(te, ne),
                               W_shell_kJ=w_dens * V_shell / 1e3, tau_build_shell_ms=w_dens * V_shell / P_loss * 1e3,
                               W_plasma_kJ=w_dens * V_plasma / 1e3, tau_build_plasma_ms=w_dens * V_plasma / P_loss * 1e3))

    # information argument: CRLB of the one-step-ahead prediction of a relaxation
    # y_k = a - b exp(-k dt / tau), observed with noise sigma_n at k = 0..r-1; predict y_r. theta = (a, b, tau)
    def pred_var(r, tau_ms, b, sn):
        tau = tau_ms / 1e3
        t = np.arange(r) * DT
        e = np.exp(-t / tau)
        J = np.stack([np.ones(r), -e, -b * e * t / tau ** 2], axis=1)
        I = J.T @ J / sn ** 2
        tp = r * DT
        ep = math.exp(-tp / tau)
        g = np.array([1.0, -ep, -b * ep * tp / tau ** 2])
        try:
            cov = np.linalg.inv(I)
        except np.linalg.LinAlgError:
            return float("inf")
        v = float(g @ cov @ g)
        return v if v > 0 else float("inf")

    info = {}
    for tau_ms in (30.0, 50.0, 80.0):
        info[str(tau_ms)] = {r: pred_var(r, tau_ms, 180.0, 39.0) / 39.0 ** 2 for r in (3, 4, 5, 6, 7, 10, 15, 30)}
    B["crlb_pred_var_over_sn2"] = info
    OUT["B"] = B
    print("B. pedestal: V_shell=%.2f m3  V_plasma=%.1f m3" % (V_shell, V_plasma))
    for c in B["cases"]:
        print("  ne=%.0e te=%4.0f ti=%4.0f tau_eq=%5.1f ms | shell W=%5.1f kJ tau_build=%5.1f ms | plasma W=%6.1f kJ tau_build=%6.1f ms"
              % (c["ne"], c["te"], c["ti"], c["tau_eq_ms"], c["W_shell_kJ"], c["tau_build_shell_ms"],
                 c["W_plasma_kJ"], c["tau_build_plasma_ms"]))
    for tau_ms, d in info.items():
        print("  CRLB pred var / sn2 at tau_rec=%s ms:" % tau_ms,
              {r: (round(v, 2) if v < 1e6 else "inf") for r, v in d.items()})


# ---------------------------------------------------------------- C. ELM-like events on the data
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


def fit_tau(curve_k, curve_y):
    """fit y(k) = yinf - amp exp(-k dt / tau) by grid on tau (ms), linear in (yinf, amp)"""
    k = np.asarray(curve_k, float)
    y = np.asarray(curve_y, float)
    ok = np.isfinite(y)
    if ok.sum() < 5:
        return float("nan"), float("nan")
    k, y = k[ok], y[ok]
    best = (np.inf, np.nan, np.nan)
    for tau in np.exp(np.linspace(math.log(3.0), math.log(2000.0), 300)):
        e = np.exp(-k * DT / (tau / 1e3))
        X = np.stack([np.ones_like(e), -e], axis=1)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        res = float(np.sum((X @ beta - y) ** 2))
        if res < best[0]:
            best = (res, tau, beta[0])
    return best[1], best[2]


def part_c(drop=0.25, post=30):
    files = sorted(DATA.glob("*.csv"))
    if not files:
        raise SystemExit("no shot CSVs found")
    events = []
    n_pairs = 0
    n_obs = 0
    n_blocks = 0
    files_with_event = set()
    for path in files:
        df = pd.read_csv(path, usecols=["time", "CES_TI"])
        t = df["time"].to_numpy(float)
        v = df["CES_TI"].to_numpy(float).copy()
        block = np.concatenate(([0], np.cumsum(np.diff(t) >= STUCK_GAP_SECONDS)))
        v[v > CUT_EV] = np.nan
        v[_stuck_mask(v, block)] = np.nan
        n_obs += int(np.isfinite(v).sum())
        n_blocks += int(block.max() + 1)
        ok = (np.isfinite(v[:-1]) & np.isfinite(v[1:]) & (block[:-1] == block[1:])
              & (np.abs(np.diff(t) - DT) < 0.5 * DT))
        n_pairs += int(ok.sum())
        idx = np.flatnonzero(ok & (v[1:] < (1 - drop) * v[:-1]) & (v[:-1] > 0)) + 1
        for i in idx:
            pre = v[i - 1]
            curve = np.full(post + 1, np.nan)
            for j in range(0, post + 1):
                if i + j < len(v) and block[i + j] == block[i] and abs(t[i + j] - t[i] - j * DT) < 0.5 * DT:
                    curve[j] = v[i + j] / pre
            events.append((path.name, int(i), float(pre), float(1 - v[i] / pre), curve))
            files_with_event.add(path.name)
    curves = np.array([e[4] for e in events])
    n_ev = len(events)
    nxt = curves[:, 1]
    sustained = np.isfinite(nxt) & (nxt < 0.9)
    single = np.isfinite(nxt) & (nxt >= 0.9)
    C = dict(files=len(files), files_with_event=len(files_with_event), blocks=n_blocks, observed_rows=n_obs,
             consecutive_pairs=n_pairs, grid_seconds=n_pairs * DT, drop_threshold=drop, events=n_ev,
             events_per_second=n_ev / (n_pairs * DT), events_sustained=int(sustained.sum()),
             events_single_point=int(single.sum()), events_next_missing=int((~np.isfinite(nxt)).sum()),
             pre_level_median=float(np.median([e[2] for e in events])),
             drop_frac_median=float(np.median([e[3] for e in events])))
    out_curves = {}
    for label, sel in (("all", np.ones(n_ev, bool)), ("sustained", sustained), ("single_point", single)):
        if sel.sum() == 0:
            continue
        med = np.nanmedian(curves[sel], axis=0)
        cnt = np.isfinite(curves[sel]).sum(axis=0)
        tau_med, yinf = fit_tau(np.arange(post + 1), med)
        taus = []
        for c in curves[sel]:
            if np.isfinite(c).sum() >= 15:
                tau_i, _ = fit_tau(np.arange(post + 1), c)
                if np.isfinite(tau_i):
                    taus.append(tau_i)
        taus = np.array(taus)
        out_curves[label] = dict(
            n=int(sel.sum()), median_curve=[float(x) for x in med], n_per_step=[int(x) for x in cnt],
            tau_rec_ms_from_median_curve=tau_med, y_inf_from_median_curve=yinf,
            per_event_tau_n=int(taus.size),
            per_event_tau_ms_q25_50_75=[float(np.quantile(taus, q)) for q in (0.25, 0.5, 0.75)] if taus.size else None,
            per_event_tau_share_under_100ms=float(np.mean(taus < 100)) if taus.size else None,
            per_event_tau_share_under_50ms=float(np.mean(taus < 50)) if taus.size else None)
    C["curves"] = out_curves
    OUT["C"] = C
    print("C. ELM-like events: files=%d (with event %d) blocks=%d obs_rows=%d consecutive_pairs=%d (%.0f s of grid)"
          % (C["files"], C["files_with_event"], C["blocks"], C["observed_rows"], C["consecutive_pairs"], C["grid_seconds"]))
    print("  events(drop>%.0f%%)=%d  rate=%.3f /s  sustained=%d single_point=%d next_missing=%d  pre_median=%.0f eV drop_median=%.2f"
          % (drop * 100, n_ev, C["events_per_second"], C["events_sustained"], C["events_single_point"],
             C["events_next_missing"], C["pre_level_median"], C["drop_frac_median"]))
    for label, d in out_curves.items():
        q = d["per_event_tau_ms_q25_50_75"]
        print("  [%s] n=%d tau_rec(median curve)=%.1f ms y_inf=%.3f | per-event tau n=%d q25/50/75=%s share<50=%s share<100=%s"
              % (label, d["n"], d["tau_rec_ms_from_median_curve"], d["y_inf_from_median_curve"], d["per_event_tau_n"],
                 None if q is None else [round(x, 1) for x in q],
                 None if d["per_event_tau_share_under_50ms"] is None else round(d["per_event_tau_share_under_50ms"], 3),
                 None if d["per_event_tau_share_under_100ms"] is None else round(d["per_event_tau_share_under_100ms"], 3)))
        print("    median curve k=0..10:", [round(x, 3) for x in d["median_curve"][:11]], " k=15,20,30:",
              [round(d["median_curve"][k], 3) for k in (15, 20, 30)])


def part_c2(drop=0.25, post=30, npre=3):
    """sensitivity: pre-level = median of the previous npre observed consecutive values (kills single-spike pre),
    and a 'recovering' class = sustained events whose k=10..30 median returns above 0.9 x pre."""
    files = sorted(DATA.glob("*.csv"))
    curves = []
    for path in files:
        df = pd.read_csv(path, usecols=["time", "CES_TI"])
        t = df["time"].to_numpy(float)
        v = df["CES_TI"].to_numpy(float).copy()
        block = np.concatenate(([0], np.cumsum(np.diff(t) >= STUCK_GAP_SECONDS)))
        v[v > CUT_EV] = np.nan
        v[_stuck_mask(v, block)] = np.nan
        n = len(v)
        for i in range(npre, n):
            w = v[i - npre:i]
            if not np.all(np.isfinite(w)) or not np.isfinite(v[i]):
                continue
            if block[i - npre] != block[i] or abs(t[i] - t[i - npre] - npre * DT) > 0.5 * DT:
                continue
            pre = float(np.median(w))
            if pre <= 0 or v[i] >= (1 - drop) * pre:
                continue
            curve = np.full(post + 1, np.nan)
            for j in range(0, post + 1):
                if i + j < n and block[i + j] == block[i] and abs(t[i + j] - t[i] - j * DT) < 0.5 * DT:
                    curve[j] = v[i + j] / pre
            curves.append(curve)
    curves = np.array(curves)
    nxt = curves[:, 1]
    sustained = np.isfinite(nxt) & (nxt < 0.9)
    late = np.nanmedian(curves[:, 10:post + 1], axis=1)
    recovering = sustained & np.isfinite(late) & (late >= 0.9)
    res = {"events": int(len(curves)), "n_sustained": int(sustained.sum()), "n_recovering": int(recovering.sum())}
    for label, sel in (("sustained", sustained), ("recovering", recovering)):
        med = np.nanmedian(curves[sel], axis=0)
        tau_med, yinf = fit_tau(np.arange(post + 1), med)
        taus = []
        for c in curves[sel]:
            if np.isfinite(c).sum() >= 15:
                tau_i, _ = fit_tau(np.arange(post + 1), c)
                if np.isfinite(tau_i):
                    taus.append(tau_i)
        taus = np.array(taus)
        res[label] = dict(n=int(sel.sum()), median_curve=[float(x) for x in med],
                          tau_rec_ms_from_median_curve=tau_med, y_inf_from_median_curve=yinf,
                          per_event_tau_n=int(taus.size),
                          per_event_tau_ms_q25_50_75=[float(np.quantile(taus, q)) for q in (0.25, 0.5, 0.75)],
                          per_event_tau_share_under_50ms=float(np.mean(taus < 50)),
                          per_event_tau_share_under_100ms=float(np.mean(taus < 100)))
        print("  [C2 %s] n=%d tau_rec(median curve)=%.1f ms y_inf=%.3f | per-event n=%d q25/50/75=%s share<50=%.3f share<100=%.3f"
              % (label, sel.sum(), tau_med, yinf, taus.size, [round(x, 1) for x in res[label]["per_event_tau_ms_q25_50_75"]],
                 res[label]["per_event_tau_share_under_50ms"], res[label]["per_event_tau_share_under_100ms"]))
        print("    median curve k=0..10:", [round(float(x), 3) for x in med[:11]], " k=15,20,30:", [round(float(med[k]), 3) for k in (15, 20, 30)])
    OUT["C2"] = res
    print("C2. pre=median of 3: events=%d sustained=%d recovering=%d" % (res["events"], res["n_sustained"], res["n_recovering"]))


if __name__ == "__main__":
    part_a()
    part_b()
    part_c()
    part_c2()
    json.dump(OUT, open(HERE / "p03_results.json", "w"), indent=1, default=float)
    print("wrote", HERE / "p03_results.json")
