"""Numerical checks for every derivation in docs/theory_foundations.md.

Each section of that document ends with a "check" line; this script is where those
lines come from. It is a data-and-theory audit in the class of
`analyze_physics_scales.py` / `analyze_noise_floor.py`: no model is trained, scored,
selected or re-selected, TEST is read only through the FROZEN comparison npz files that
THESIS_RESULTS.md already quotes, and nothing here changes a verdict.

Parts
-----
A  CES photon statistics: Cramer-Rao bounds of a Gaussian line fit (Monte Carlo vs the
   closed form) and the photon count implied by the measured T_i / V_rot noise floors.
B  Transport scales: the exact NRL equipartition rate vs the (m_i/2m_e)*tau_e convention,
   NBI torque per MW, the radial-force-balance term sizes.
C  Mirnov decimation: lag-1 autocorrelation of a kHz mode sampled at 100 Hz (analytic
   vs Monte Carlo) and the amplitude information one sample carries.
D  Ornstein-Uhlenbeck + measurement-noise model FITTED TO THE 641 FILES (robust
   semivariogram), then every closed-form prediction that model makes: persistence /
   interpolation / causal-filter error ratios, the value of history length K, the
   filter's memory length, the gap at which interpolation overtakes causal prediction.
E  Identities on the frozen TEST artifacts: the MSE skill as a quantile integral (tail
   effect), the spike-dilution formula, the ensemble disagreement identity, the cluster
   design effect, PCHIP-vs-linear noise amplification, the share of val MSE a bounded
   loss can see.

Run from the repo root:  py ces_prediction/analyze_foundations.py
Writes data/.foundations.json.  Console output is ASCII only (cp949 console).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baselines_interpolation import predict_linear, predict_pchip  # noqa: E402

RNG = np.random.default_rng(20261013)
STUCK_GAP_SECONDS = 0.5
CUT_EV = 3000.0
DT = 0.010  # grid step [s]

# physical constants (SI)
E_CHARGE = 1.602176634e-19
AMU = 1.66053906660e-27
M_E = 9.1093837015e-31
C_LIGHT = 2.99792458e8
EPS0 = 8.8541878128e-12

OUT: dict = {}


def data_dir() -> Path:
    return Path(os.environ.get("CES_DATA_DIR", "data"))


def hr(title):
    print("")
    print("=" * 78)
    print(title)
    print("=" * 78)


# =============================================================================
# PART A -- CES photon statistics
# =============================================================================
def part_a():
    hr("PART A -- CES line fit: Cramer-Rao bounds and the implied photon count")
    lam0 = 529.05e-9          # C VI n=8->7 [m]
    m_c = 12.0 * AMU          # carbon-12
    ti_ev = 592.8             # observed CES_TI median (cut population, S8aq)
    vth = np.sqrt(ti_ev * E_CHARGE / m_c)            # sqrt(kT/m) [m/s]
    sigma_lam = lam0 * vth / C_LIGHT                  # Doppler 1-sigma width [m]
    fwhm = 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma_lam
    print("C VI 529.05 nm, T_i = %.0f eV: v_th = sqrt(kT/m_C) = %.1f km/s" % (ti_ev, vth / 1e3))
    print("  Doppler sigma_lambda = %.4f nm, FWHM = %.4f nm" % (sigma_lam * 1e9, fwhm * 1e9))

    # Closed-form CRB for N photons (no background): sigma_mu = s/sqrt(N), sigma_s = s/sqrt(2N)
    # -> sigma_v = v_th/sqrt(N),  sigma_T/T = 2*sigma_s/s = sqrt(2/N)
    # Implied photon counts from the measured bulk noise floors (S8aq MAD):
    mad_ti, mad_vt = 46.4, 4.3  # eV, km/s
    n_from_ti = 2.0 / (mad_ti / ti_ev) ** 2
    n_from_vt = (vth / 1e3 / mad_vt) ** 2
    print("  measured bulk floors: T_i MAD %.1f eV (%.1f%% rel), V_rot MAD %.1f km/s"
          % (mad_ti, 100 * mad_ti / ti_ev, mad_vt))
    print("  photon count implied by T_i floor  : N = 2/(sigma_T/T)^2 = %.0f" % n_from_ti)
    print("  photon count implied by V_rot floor: N = (v_th/sigma_v)^2  = %.0f" % n_from_vt)
    print("  ratio N_T / N_V = %.2f  (1.0 = the two floors are one photon budget)"
          % (n_from_ti / n_from_vt))

    # Monte Carlo: Poisson spectrum, 4-parameter Gaussian fit, compare scatter with CRB.
    from scipy.optimize import minimize

    pix = np.linspace(-4.0, 4.0, 48)  # in units of true sigma
    shape = np.exp(-0.5 * pix ** 2)
    shape /= shape.sum()

    def nll(theta, counts):
        a, mu, sg, b = theta
        lam = a * np.exp(-0.5 * ((pix - mu) / sg) ** 2) + b
        lam = np.clip(lam, 1e-9, None)
        return float(np.sum(lam - counts * np.log(lam)))

    mc = {}
    for n_phot, n_trials in ((300, 1500), (3000, 600)):
        mus, sigs = [], []
        for _ in range(n_trials):
            counts = RNG.poisson(n_phot * shape).astype(float)
            r = minimize(nll, x0=[counts.max(), 0.0, 1.0, 0.0], args=(counts,), method="Nelder-Mead",
                         options={"xatol": 1e-7, "fatol": 1e-9, "maxiter": 6000})
            if not r.success:
                continue
            mus.append(r.x[1])
            sigs.append(abs(r.x[2]))
        mus, sigs = np.array(mus), np.array(sigs)
        crb_mu = 1.0 / np.sqrt(n_phot)
        crb_s = 1.0 / np.sqrt(2.0 * n_phot)
        # 4-parameter Fisher matrix (a, mu, s, b) -> the CRB a free-background fit faces
        lam = n_phot * shape + 1e-9
        g = n_phot * shape
        J = np.vstack([shape, g * pix, g * pix ** 2, np.ones_like(pix)])
        fisher = (J / lam) @ J.T
        cov = np.linalg.inv(fisher)
        crb4 = np.sqrt(np.diag(cov))
        corr_fisher = cov[1, 2] / (crb4[1] * crb4[2])
        mad_mu = 1.4826 * np.median(np.abs(mus - np.median(mus)))
        mad_s = 1.4826 * np.median(np.abs(sigs - np.median(sigs)))
        print("  Monte Carlo N=%d photons, %d Poisson-ML fits (free background):" % (n_phot, len(mus)))
        print("    center: std/CRB4 = %.3f (robust %.3f)   width: std/CRB4 = %.3f (robust %.3f)   corr(center,width) = %+.3f"
              % (mus.std() / crb4[1], mad_mu / crb4[1], sigs.std() / crb4[2], mad_s / crb4[2],
                 np.corrcoef(mus, sigs)[0, 1]))
        print("    CRB4/CRB2 (cost of fitting a background): center x%.2f, width x%.2f; Fisher corr(center,width) = %+.1e"
              % (crb4[1] / crb_mu, crb4[2] / crb_s, corr_fisher))
        mc[n_phot] = {"trials": int(len(mus)), "std_center_over_crb4": float(mus.std() / crb4[1]),
                      "robust_center_over_crb4": float(mad_mu / crb4[1]),
                      "std_width_over_crb4": float(sigs.std() / crb4[2]),
                      "robust_width_over_crb4": float(mad_s / crb4[2]),
                      "corr_center_width": float(np.corrcoef(mus, sigs)[0, 1]),
                      "center_inflation": float(crb4[1] / crb_mu), "width_inflation": float(crb4[2] / crb_s),
                      "fisher_corr_center_width": float(corr_fisher)}
    OUT["A"] = {
        "line_nm": lam0 * 1e9, "ti_ev": ti_ev, "vth_km_s": vth / 1e3,
        "doppler_sigma_nm": sigma_lam * 1e9, "fwhm_nm": fwhm * 1e9,
        "n_photons_from_ti_floor": n_from_ti, "n_photons_from_vt_floor": n_from_vt,
        "mc": mc,
    }


# =============================================================================
# PART B -- transport scales
# =============================================================================
def part_b():
    hr("PART B -- transport: equipartition rate, NBI torque per MW, radial force balance")
    ln_lam = 17.0
    m_i = 2.0 * AMU
    rows = []
    for te in (500.0, 1000.0, 2000.0):
        for ne in (2e19, 3e19, 5e19):
            # NRL exact (cgs): nu_eps^{i|e} = 1.8e-19 sqrt(m_e m_i) Z^2 n_e lnL / (m_i T_e + m_e T_i)^1.5
            me_g, mi_g, ne_cm = M_E * 1e3, m_i * 1e3, ne * 1e-6
            ti = te
            nu = 1.8e-19 * np.sqrt(me_g * mi_g) * ne_cm * ln_lam / (mi_g * te + me_g * ti) ** 1.5
            tau_exact = 1.0 / nu
            tau_e = 3.44e5 * te ** 1.5 / (ne_cm * ln_lam)
            tau_conv = (m_i / (2.0 * M_E)) * tau_e
            rows.append((te, ne, tau_exact * 1e3, tau_conv * 1e3, tau_conv / tau_exact))
    print("%8s %10s %14s %18s %8s" % ("Te[eV]", "ne[m-3]", "tau_eq NRL[ms]", "(mi/2me)tau_e[ms]", "ratio"))
    for te, ne, a, b, r in rows:
        print("%8.0f %10.0e %14.1f %18.1f %8.2f" % (te, ne, a, b, r))

    # NBI torque per unit power: ions arrive at P/E_b per second, each carrying m v_b R_tan,
    # so T/P = m v_b R_tan / E_b = 2 R_tan / v_b with v_b = sqrt(2 E_b / m_D)
    eb_kev, r_tan = 100.0, 1.5
    v_b = np.sqrt(2.0 * eb_kev * 1e3 * E_CHARGE / m_i)
    t_per_mw = 2.0 * r_tan / v_b * 1e6
    print("")
    print("NBI (assumed E_b = %.0f keV D, R_tan = %.2f m): v_b = %.2e m/s, torque/power = %.3f N m per MW"
          % (eb_kev, r_tan, v_b, t_per_mw))
    print("  at fixed power, torque scales as R_tan * E_b^-1/2 -> 'power != torque'")

    # radial force balance term sizes (assumed pedestal numbers)
    n_i, ti_ev, grad_len = 3e19, 600.0, 0.05
    grad_p_term = n_i * ti_ev * E_CHARGE / grad_len / (E_CHARGE * n_i)  # [V/m]
    v_phi, b_theta = 1.0e5, 0.3
    vphi_term = v_phi * b_theta
    print("radial force balance E_r = (1/Zen) dp_i/dr - v_theta B_phi + v_phi B_theta:")
    print("  |(1/en) dp_i/dr| ~ %.1f kV/m   |v_phi B_theta| ~ %.1f kV/m  (same order: v_phi not readable from T_i)"
          % (grad_p_term / 1e3, vphi_term / 1e3))
    # Linear response check: T_i relaxes toward T_e on tau_eq while T_e wanders on tau_c.
    # Prediction: corr(T_e, T_i) > corr(T_e, dT_i) > 0, which is the ordering S8ar measured
    # (+0.31 level vs +0.078 rate for ECEI).
    tau_eq, tau_c, n_steps = 0.040, 0.160, 200000
    rho_c = np.exp(-DT / tau_c)
    te = np.zeros(n_steps)
    ti = np.zeros(n_steps)
    for i in range(1, n_steps):
        te[i] = rho_c * te[i - 1] + RNG.normal() * np.sqrt(1 - rho_c ** 2)
        ti[i] = ti[i - 1] + (DT / tau_eq) * (te[i - 1] - ti[i - 1]) + RNG.normal() * 0.05
    noise = RNG.normal(size=n_steps) * 1.0            # measurement noise on both, unit variance
    te_obs, ti_obs = te + noise, ti + RNG.normal(size=n_steps) * 1.0
    r_level = np.corrcoef(te_obs[1:], ti_obs[1:])[0, 1]
    r_rate = np.corrcoef(te_obs[1:], np.diff(ti_obs))[0, 1]
    print("linear response (tau_eq %.0f ms driven by tau_c %.0f ms, unit noise on both): corr level %.3f, corr rate %.3f"
          % (tau_eq * 1e3, tau_c * 1e3, r_level, r_rate))
    OUT["B"] = {
        "linear_response_sim": {"tau_eq_ms": tau_eq * 1e3, "tau_c_ms": tau_c * 1e3,
                                "corr_level": float(r_level), "corr_rate": float(r_rate)},
        "tau_eq_rows": [{"te_ev": r[0], "ne_m3": r[1], "tau_nrl_ms": r[2], "tau_conv_ms": r[3],
                         "ratio_conv_over_nrl": r[4]} for r in rows],
        "nbi": {"eb_kev": eb_kev, "r_tan_m": r_tan, "v_b_m_s": v_b, "torque_Nm_per_MW": t_per_mw},
        "force_balance_kV_m": {"grad_p_term": grad_p_term / 1e3, "vphi_btheta_term": vphi_term / 1e3},
    }


# =============================================================================
# PART C -- Mirnov decimation
# =============================================================================
def part_c():
    hr("PART C -- Mirnov: a kHz mode sampled at 100 Hz without an anti-aliasing filter")
    f0, dur, fs = 4000.0, 40.0, 200000.0
    t = np.arange(0.0, dur, 1.0 / fs)
    # frequency wander: random walk with ~5 % rms excursion, i.e. ~200 Hz
    rows = []
    for sig_f in (0.0, 50.0, 100.0, 200.0, 400.0):
        # instantaneous frequency: f0 + OU jitter with std sig_f, correlation 50 ms
        jit = np.zeros_like(t)
        a = np.exp(-(1.0 / fs) / 0.05)
        eps = RNG.normal(size=t.size) * sig_f * np.sqrt(1 - a * a)
        for i in range(1, t.size):
            jit[i] = a * jit[i - 1] + eps[i]
        phase = 2 * np.pi * np.cumsum(f0 + jit) / fs
        env = 1.0 + 0.5 * np.sin(2 * np.pi * 0.7 * t)      # slow amplitude envelope
        x = env * np.cos(phase)
        dec = x[:: int(fs * DT)]                            # 100 Hz decimation, no filter
        r1 = np.corrcoef(dec[:-1], dec[1:])[0, 1]
        r1_hi = np.corrcoef(x[:-1], x[1:])[0, 1]
        # analytic: Gaussian spectral width sig_f -> r1 = cos(2 pi f0 dt) exp(-2 pi^2 sig_f^2 dt^2)
        r1_an = np.cos(2 * np.pi * f0 * DT) * np.exp(-2 * np.pi ** 2 * sig_f ** 2 * DT ** 2)
        # single-sample amplitude information: relative std of x^2 as an estimator of A^2/2
        env_d = env[:: int(fs * DT)]
        pw = dec ** 2 / (env_d ** 2 / 2.0)                  # single-sample estimate of the true power
        rel = np.std(pw) / np.mean(pw)                      # theory: sqrt(1/2) = 0.707
        # 10 ms window RMS from the full-rate stream
        nwin = int(fs * DT)
        rms = np.sqrt(np.mean(x[: (x.size // nwin) * nwin].reshape(-1, nwin) ** 2, axis=1))
        env_dec = env[: (x.size // nwin) * nwin].reshape(-1, nwin).mean(axis=1) / np.sqrt(2)
        rel_win = np.std(rms / env_dec) / np.mean(rms / env_dec)
        rows.append((sig_f, r1_hi, r1, r1_an, rel, rel_win))
    print("%10s %12s %14s %14s %16s %16s" % ("sig_f[Hz]", "r1 @200kHz", "r1 @100Hz MC", "r1 analytic",
                                             "rel.err 1 sample", "rel.err 10ms RMS"))
    for r in rows:
        print("%10.0f %12.3f %14.3f %14.3f %16.2f %16.3f" % r)
    print("  grid 10 ms -> Nyquist 50 Hz; any spectral width >~ 1/(2 pi dt) = %.0f Hz erases lag-1 correlation"
          % (1 / (2 * np.pi * DT)))
    OUT["C"] = [{"sig_f_hz": r[0], "r1_fullrate": r[1], "r1_decimated_mc": r[2], "r1_analytic": r[3],
                 "rel_err_single_sample_power": r[4], "rel_err_10ms_rms": r[5]} for r in rows]


# =============================================================================
# PART D -- OU + noise model fitted to the real series
# =============================================================================
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


def load_targets():
    files = sorted(data_dir().glob("*.csv"))
    if not files:
        raise SystemExit("no shot CSVs found -- point CES_DATA_DIR at the real folder")
    series = {"CES_TI": [], "CES_VT": []}
    for path in files:
        df = pd.read_csv(path, usecols=["time", "CES_TI", "CES_VT"])
        time = df["time"].to_numpy(dtype=float)
        block = np.concatenate(([0], np.cumsum(np.diff(time) >= STUCK_GAP_SECONDS)))
        for col in series:
            v = df[col].to_numpy(dtype=float).copy()
            if col == "CES_TI":
                v[v > CUT_EV] = np.nan
            v[_stuck_mask(v, block)] = np.nan
            series[col].append((time, block, v))
    return files, series


def robust_variogram(series, max_lag=12):
    """gamma(h) from pairs h grid steps apart inside one block, both observed.
    Returns classical 0.5*mean(d^2) and robust 0.5*(1.4826*MAD(d))^2."""
    out = {}
    for h in range(1, max_lag + 1):
        ds = []
        for time, block, v in series:
            if v.size <= h:
                continue
            a, b = v[:-h], v[h:]
            same = block[:-h] == block[h:]
            step = np.abs((time[h:] - time[:-h]) - h * DT) < 0.5 * DT
            m = same & step & np.isfinite(a) & np.isfinite(b)
            if m.any():
                ds.append(b[m] - a[m])
        d = np.concatenate(ds)
        med = np.median(d)
        mad = 1.4826 * np.median(np.abs(d - med))
        out[h] = {"n": int(d.size), "classical": 0.5 * float(np.mean(d * d)),
                  "robust": 0.5 * float(mad * mad)}
    return out


def fit_ou(gamma, key, lags):
    """gamma(h) = s_n^2 + s_s^2 (1 - exp(-h dt / tau)); grid over tau, linear LS for the two
    variances (clipped at 0). Returns (s_n2, s_s2, tau, rmse)."""
    h = np.array(lags, dtype=float)
    g = np.array([gamma[k][key] for k in lags])
    best = None
    for tau in np.geomspace(0.005, 5.0, 400):
        x = 1.0 - np.exp(-h * DT / tau)
        design = np.vstack([np.ones_like(x), x]).T
        coef, *_ = np.linalg.lstsq(design, g, rcond=None)
        coef = np.clip(coef, 0.0, None)
        resid = float(np.sqrt(np.mean((design @ coef - g) ** 2)))
        if best is None or resid < best[3]:
            best = (float(coef[0]), float(coef[1]), float(tau), resid)
    return best


def ou_predictions(s_n2, s_s2, tau, unit):
    """Every closed-form consequence of a noisy OU series on a 10 ms grid."""
    rho = np.exp(-DT / tau)
    res = {"rho_1step": rho, "a": DT / tau}

    def cov_mat(lags_i, lags_j):
        d = np.abs(np.subtract.outer(lags_i, lags_j)).astype(float)
        return s_s2 * rho ** d

    def best_linear(past_lags, future_lags=()):
        """MSE of the best linear predictor of the NOISY target at lag 0 from noisy
        observations at the given lags (positive = past, negative = future)."""
        lags = np.array(list(past_lags) + list(future_lags), dtype=float)
        sig = cov_mat(lags, lags) + s_n2 * np.eye(lags.size)
        c = cov_mat(lags, np.array([0.0]))[:, 0]
        w = np.linalg.solve(sig, c)
        return float(s_s2 + s_n2 - c @ w), w

    # persistence vs interpolation vs filter, all at one-step gaps
    mse_pers = 2 * s_n2 + 2 * s_s2 * (1 - rho)
    mse_interp_fixed = 1.5 * s_n2 + s_s2 * (1.5 + 0.5 * rho ** 2 - 2 * rho)   # equal-weight 2 nbrs
    mse_interp_opt, _ = best_linear([1], [-1])
    mse_k = {}
    for k in (1, 2, 3, 5, 10, 30, 100):
        mse_k[k], _ = best_linear(range(1, k + 1))
    # steady-state Kalman (K -> inf) via Riccati
    p = s_s2
    for _ in range(10000):
        p_minus = rho ** 2 * p + s_s2 * (1 - rho ** 2)
        kg = p_minus / (p_minus + s_n2)
        p = (1 - kg) * p_minus
    mse_inf = p_minus + s_n2
    mem_steps = 1.0 / (-np.log(rho * (1 - kg)))
    res.update({
        "mse_persistence": mse_pers, "mse_interp_equal": mse_interp_fixed,
        "mse_interp_opt": mse_interp_opt, "mse_filter_K": mse_k, "mse_filter_inf": mse_inf,
        "kalman_gain": kg, "filter_memory_steps": mem_steps,
        "rmse_ratio_interp_over_pers": np.sqrt(mse_interp_opt / mse_pers),
        "rmse_ratio_filter_over_pers": np.sqrt(mse_inf / mse_pers),
        "rmse_ratio_last1_over_pers": np.sqrt(mse_k[1] / mse_pers),
        "skill_filter_vs_pers": 1 - mse_inf / mse_pers,
        "skill_last1_vs_pers": 1 - mse_k[1] / mse_pers,
        "skill_interp_vs_pers": 1 - mse_interp_opt / mse_pers,
        "skill_filter_vs_interp": 1 - mse_inf / mse_interp_opt,
        "noise_free_skill_filter_vs_pers_small_a": DT / tau / 2,
    })
    # gap dependence: past anchor g steps back (all older past available), future nbr 1 step ahead
    gaps = {}
    for g in (1, 2, 3, 5, 8, 12, 20):
        past = list(range(g, g + 30))
        m_c, _ = best_linear(past)
        m_i, _ = best_linear(past, [-1])
        m_p = 2 * s_n2 + 2 * s_s2 * (1 - rho ** g)
        gaps[g] = {"gap_ms": g * 10, "mse_causal": m_c, "mse_interp": m_i, "mse_persistence": m_p,
                   "skill_causal_vs_interp": 1 - m_c / m_i, "skill_causal_vs_pers": 1 - m_c / m_p}
    res["by_gap"] = gaps
    # how much the K-th past observation still adds (Markov + noise)
    res["gain_from_K_over_1_pct"] = {k: 100 * (1 - mse_k[k] / mse_k[1]) for k in mse_k}
    # print
    print("  fitted: sigma_n = %.2f %s, sigma_s = %.2f %s, tau = %.0f ms (rho = %.3f per step)"
          % (np.sqrt(s_n2), unit, np.sqrt(s_s2), unit, tau * 1e3, rho))
    print("  one-step-gap RMSE: persistence %.2f | interp(2 nbrs, opt) %.2f | filter(last 1) %.2f | filter(inf) %.2f %s"
          % (np.sqrt(mse_pers), np.sqrt(mse_interp_opt), np.sqrt(mse_k[1]), np.sqrt(mse_inf), unit))
    print("  RMSE ratios vs persistence: interp %.3f | filter %.3f | last-1 %.3f"
          % (res["rmse_ratio_interp_over_pers"], res["rmse_ratio_filter_over_pers"],
             res["rmse_ratio_last1_over_pers"]))
    print("  skill vs persistence: filter %.3f (noise-free limit a/2 = %.3f) | interp %.3f ; filter vs interp %.3f"
          % (res["skill_filter_vs_pers"], res["noise_free_skill_filter_vs_pers_small_a"],
             res["skill_interp_vs_pers"], res["skill_filter_vs_interp"]))
    print("  value of history: MSE gain over last-1 by K =", {k: round(v, 2) for k, v in res["gain_from_K_over_1_pct"].items()}, "%")
    print("  Kalman gain %.3f, filter memory %.1f steps = %.0f ms" % (kg, mem_steps, mem_steps * 10))
    print("  gap dependence (history-only causal vs 2-sided interp):",
          {g: round(v["skill_causal_vs_interp"], 3) for g, v in gaps.items()})
    return res


def part_d():
    hr("PART D -- noisy Ornstein-Uhlenbeck model fitted to the 641 files (held-free, 3 keV cut)")
    files, series = load_targets()
    print("files: %d" % len(files))
    OUT["D"] = {}
    for col, unit in (("CES_TI", "eV"), ("CES_VT", "km/s")):
        gam = robust_variogram(series[col])
        lags = list(range(1, 13))
        print("")
        print("%s semivariogram sqrt(gamma(h)) [%s]  (classical / robust):" % (col, unit))
        print("   " + " ".join("h%-2d" % h for h in lags))
        print("   " + " ".join("%4.0f" % np.sqrt(gam[h]["classical"]) for h in lags))
        print("   " + " ".join("%4.1f" % np.sqrt(gam[h]["robust"]) for h in lags))
        fits = {}
        for key in ("robust", "classical"):
            s_n2, s_s2, tau, rmse = fit_ou(gam, key, lags)
            print("  [%s fit]" % key)
            pred = ou_predictions(s_n2, s_s2, tau, unit)
            fits[key] = {"sigma_n": float(np.sqrt(s_n2)), "sigma_s": float(np.sqrt(s_s2)),
                         "tau_ms": tau * 1e3, "fit_rmse": rmse, "pred": pred}
        OUT["D"][col] = {"variogram": {h: gam[h] for h in lags}, "fits": fits}


# =============================================================================
# PART E -- identities on the frozen TEST artifacts
# =============================================================================
def part_e():
    hr("PART E -- identities checked on the frozen TEST artifacts (seed 42, cut population)")
    cut = np.load(data_dir() / ".b1_seqv2_s42_i42" / "comparison_errors_test.npz")
    incl = np.load(data_dir() / ".b5i_seqv2_s42" / "comparison_errors_test.npz")
    E = {}

    # --- E1 tail identity: skill = 1 - sum se_m / sum se_b, split by baseline-error quantile
    for tgt in ("CES_TI", "CES_VT"):
        sm, sb = cut[f"{tgt}_se_model"].astype(float), cut[f"{tgt}_se_pchip"].astype(float)
        skill = 1 - sm.sum() / sb.sum()
        eb = np.sqrt(sb)
        order = np.argsort(sm + sb)                       # symmetric: no selection on either arm
        cum = np.cumsum((sb - sm)[order]) / sb.sum()      # running contribution to skill
        qs = {}
        for q in (0.5, 0.7, 0.75, 0.9, 0.99, 1.0):
            idx = int(np.ceil(q * len(order))) - 1
            qs[q] = float(cum[idx])
        share_top1 = float(sb[order][-int(0.01 * len(order)):].sum() / sb.sum())
        q_ratio = {q: float(np.quantile(np.sqrt(sm), q) / np.quantile(eb, q)) for q in (0.5, 0.7, 0.75, 0.9, 0.99)}
        med_m, med_b = np.median(np.sqrt(sm)), np.median(eb)
        print("%s: MSE skill vs PCHIP %+.4f; running skill contribution up to (se_m+se_b) quantile:"
              % (tgt, skill))
        print("   " + "  ".join("q%.2f:%+.3f" % (q, v) for q, v in qs.items()))
        print("   median AE model %.1f vs PCHIP %.1f; top-1%% rows carry %.1f%% of PCHIP SSE"
              % (med_m, med_b, 100 * share_top1))
        print("   |e| quantile ratio model/PCHIP:", {q: round(v, 2) for q, v in q_ratio.items()})
        E[f"tail_{tgt}"] = {"skill": float(skill), "running_contribution": qs,
                            "median_model": float(med_m), "median_pchip": float(med_b),
                            "top1pct_share_pchip_sse": share_top1, "quantile_ratio": q_ratio}

    # --- E2 spike dilution on the inclusive population: skill' = skill/(1-f) when the removed
    #     rows cost both arms the same
    y = incl["CES_TI_y_true"].astype(float)
    sm, sb = incl["CES_TI_se_model"].astype(float), incl["CES_TI_se_pchip"].astype(float)
    keep = y <= CUT_EV
    skill_all = 1 - sm.sum() / sb.sum()
    skill_cut = 1 - sm[keep].sum() / sb[keep].sum()
    f_b = sb[~keep].sum() / sb.sum()
    f_m = sm[~keep].sum() / sm.sum()
    pred_equal = skill_all / (1 - f_b)
    print("")
    print("inclusive s42 T_i: skill %+.4f -> %+.4f after removing %d rows > 3 keV (%.2f%% of rows)"
          % (skill_all, skill_cut, int((~keep).sum()), 100 * (~keep).mean()))
    pred_exact = 1 - (sm.sum() * (1 - f_m)) / (sb.sum() * (1 - f_b))
    print("   removed rows carry %.1f%% of PCHIP SSE and %.1f%% of model SSE; equal-cost formula skill/(1-f) = %+.4f, exact 1-(1-f_m)SSE_m/((1-f_b)SSE_b) = %+.4f"
          % (100 * f_b, 100 * f_m, pred_equal, pred_exact))
    E["spike_dilution"] = {"skill_all": float(skill_all), "skill_after_cut": float(skill_cut),
                           "f_pchip": float(f_b), "f_model": float(f_m), "formula_equal_cost": float(pred_equal),
                           "rows_removed": int((~keep).sum())}

    # --- E3 ensemble identity: MSE_ens = mean_i MSE_i - mean_rows Var_i(yhat_i)
    ens = np.load(data_dir() / ".b15_ens4_cut_s42" / "comparison_errors_test.npz")
    members = [np.load(data_dir() / f".b1_seqv2_s42_i{i}" / "comparison_errors_test.npz")
               for i in (42, 1, 7, 123)]
    for tgt in ("CES_TI", "CES_VT"):
        for m in members:
            assert np.array_equal(m[f"{tgt}_shot"], ens[f"{tgt}_shot"])
            assert np.allclose(m[f"{tgt}_dt_ms"], ens[f"{tgt}_dt_ms"])
            assert np.allclose(m[f"{tgt}_se_pchip"], ens[f"{tgt}_se_pchip"])
        se_m = np.stack([m[f"{tgt}_se_model"].astype(float) for m in members])
        se_e = ens[f"{tgt}_se_model"].astype(float)
        sb = ens[f"{tgt}_se_pchip"].astype(float)
        jensen_ok = float(np.mean(se_e <= se_m.mean(axis=0) * (1 + 1e-4) + 1e-6))
        disagreement = se_m.mean(axis=0).mean() - se_e.mean()      # = mean row variance of members
        gain_vs_mean_member = disagreement / sb.mean()
        gain_vs_canonical = (se_m[0].mean() - se_e.mean()) / sb.mean()
        print("ensemble (%s): Jensen row-wise se_ens <= mean se_i holds on %.2f%% rows; "
              "mean member disagreement var = %.1f (= %.4f of PCHIP MSE); skill gain vs canonical %+.4f"
              % (tgt, 100 * jensen_ok, disagreement, gain_vs_mean_member, gain_vs_canonical))
        E[f"ensemble_{tgt}"] = {"jensen_fraction": jensen_ok, "disagreement_var": float(disagreement),
                                "gain_vs_mean_member": float(gain_vs_mean_member),
                                "gain_vs_canonical": float(gain_vs_canonical)}

    # --- E4 design effect of shot clustering on the paired difference
    for tgt in ("CES_TI", "CES_VT"):
        d = cut[f"{tgt}_se_pchip"].astype(float) - cut[f"{tgt}_se_model"].astype(float)
        shot = cut[f"{tgt}_shot"]
        n = d.size
        se_naive = d.std(ddof=1) / np.sqrt(n)
        shots, inv = np.unique(shot, return_inverse=True)
        sums = np.bincount(inv, weights=d)
        cnts = np.bincount(inv).astype(float)
        resid = sums - d.mean() * cnts
        se_cl = np.sqrt(np.sum(resid ** 2) * len(shots) / (len(shots) - 1)) / n
        deff = (se_cl / se_naive) ** 2
        # intra-shot correlation implied by deff = 1 + (m-1) rho
        m_bar = n / len(shots)
        rho = (deff - 1) / (m_bar - 1)
        print("%s paired diff: n=%d rows, %d shots, SE naive %.1f vs cluster %.1f -> design effect %.1f, "
              "n_eff = %.0f, implied intra-shot rho = %.3f"
              % (tgt, n, len(shots), se_naive, se_cl, deff, n / deff, rho))
        E[f"deff_{tgt}"] = {"n": int(n), "shots": int(len(shots)), "se_naive": float(se_naive),
                            "se_cluster": float(se_cl), "design_effect": float(deff),
                            "n_eff": float(n / deff), "rho_intra": float(rho)}

    # --- E5 PCHIP vs linear: measured ratio on the artifacts + Monte Carlo on the fitted OU
    for tgt in ("CES_TI", "CES_VT"):
        sp, sl = cut[f"{tgt}_se_pchip"].astype(float), cut[f"{tgt}_se_linear"].astype(float)
        E[f"pchip_over_linear_rmse_{tgt}"] = float(np.sqrt(sp.mean() / sl.mean()))
        print("%s measured RMSE ratio PCHIP/linear = %.4f" % (tgt, E[f"pchip_over_linear_rmse_{tgt}"]))
    fit = OUT["D"]["CES_TI"]["fits"]["robust"]
    s_n, s_s, tau = fit["sigma_n"], fit["sigma_s"], fit["tau_ms"] / 1e3
    rho = np.exp(-DT / tau)
    n_sim, n_len = 3000, 9
    err_l, err_p = [], []
    for _ in range(n_sim):
        x = np.zeros(n_len)
        x[0] = RNG.normal() * s_s
        for i in range(1, n_len):
            x[i] = rho * x[i - 1] + RNG.normal() * s_s * np.sqrt(1 - rho * rho)
        yobs = x + RNG.normal(size=n_len) * s_n
        times = np.arange(n_len) * DT
        k = n_len // 2
        nb = np.delete(np.arange(n_len), k)
        err_l.append(predict_linear(times[nb], yobs[nb], times[k]) - yobs[k])
        err_p.append(predict_pchip(times[nb], yobs[nb], times[k]) - yobs[k])
    ratio_mc = float(np.sqrt(np.mean(np.square(err_p)) / np.mean(np.square(err_l))))
    print("Monte Carlo on the fitted T_i OU+noise model: RMSE PCHIP/linear = %.4f" % ratio_mc)
    E["pchip_over_linear_rmse_mc"] = ratio_mc

    # --- E6 what share of the (val-like) MSE a bounded loss can see: rows with |e| <= delta
    for tgt, delta in (("CES_TI", 129.9), ("CES_VT", 16.4)):
        sm = cut[f"{tgt}_se_model"].astype(float)
        inside = np.sqrt(sm) <= delta
        share = sm[inside].sum() / sm.sum()
        print("%s: rows with |e| <= delta=%.1f are %.1f%% of rows but %.1f%% of model SSE"
              % (tgt, delta, 100 * inside.mean(), 100 * share))
        E[f"bulk_share_{tgt}"] = {"delta": delta, "row_fraction": float(inside.mean()), "sse_share": float(share)}
    # inclusive population, same question
    sm = incl["CES_TI_se_model"].astype(float)
    inside = np.sqrt(sm) <= 274.3
    print("inclusive T_i: rows with |e| <= 274.3 eV are %.1f%% of rows, %.1f%% of model SSE"
          % (100 * inside.mean(), 100 * sm[inside].sum() / sm.sum()))
    E["bulk_share_incl_CES_TI"] = {"delta": 274.3, "row_fraction": float(inside.mean()),
                                   "sse_share": float(sm[inside].sum() / sm.sum())}

    # --- E7 one-step population (dt <= 15 ms) ladder on the artifacts, to compare with Part D
    for tgt in ("CES_TI", "CES_VT"):
        dt = cut[f"{tgt}_dt_ms"]
        m = dt <= 15.0
        lad = {}
        pers = cut[f"{tgt}_se_persistence"][m].astype(float)
        for arm in ("model", "gp_causal", "gp", "pchip", "linear", "persistence"):
            se = cut[f"{tgt}_se_{arm}"][m].astype(float)
            lad[arm] = {"rmse": float(np.sqrt(se.mean())), "median_ae": float(np.median(np.sqrt(se))),
                        "skill_vs_pers": float(1 - se.mean() / pers.mean()),
                        "median_ratio_vs_pers": float(np.median(np.sqrt(se)) / np.median(np.sqrt(pers)))}
        print("%s, dt <= 15 ms (n=%d):" % (tgt, int(m.sum())))
        for arm, v in lad.items():
            print("   %-12s RMSE %7.1f  medAE %6.1f  skill vs pers %+.3f  median ratio vs pers %.3f"
                  % (arm, v["rmse"], v["median_ae"], v["skill_vs_pers"], v["median_ratio_vs_pers"]))
        E[f"ladder_1step_{tgt}"] = lad
        # by-gap: causal GP vs PCHIP
        bins = [(0, 15), (15, 25), (25, 45), (45, 105), (105, 1e9)]
        bg = {}
        for lo, hi in bins:
            mm = (dt > lo) & (dt <= hi)
            if mm.sum() < 20:
                continue
            sc = cut[f"{tgt}_se_gp_causal"][mm].astype(float)
            sp = cut[f"{tgt}_se_pchip"][mm].astype(float)
            sx = cut[f"{tgt}_se_model"][mm].astype(float)
            spers = cut[f"{tgt}_se_persistence"][mm].astype(float)
            fallback = float(np.mean(np.isclose(sp, spers, rtol=1e-9, atol=1e-9)))
            med_ratio_gp = float(np.median(np.sqrt(sc)) / np.median(np.sqrt(sp)))
            med_ratio_model = float(np.median(np.sqrt(sx)) / np.median(np.sqrt(sp)))
            bg[f"({lo},{hi}]"] = {"n": int(mm.sum()), "gp_causal_vs_pchip": float(1 - sc.mean() / sp.mean()),
                                   "model_vs_pchip": float(1 - sx.mean() / sp.mean()),
                                   "pchip_fallback_to_persistence": fallback,
                                   "median_ratio_gp_causal_over_pchip": med_ratio_gp,
                                   "median_ratio_model_over_pchip": med_ratio_model}
        print("   by gap, skill vs PCHIP (causal GP / model / PR2 fallback share):",
              {k: (round(v["gp_causal_vs_pchip"], 2), round(v["model_vs_pchip"], 2),
                   round(v["pchip_fallback_to_persistence"], 2)) for k, v in bg.items()})
        print("   by gap, MEDIAN |e| ratio vs PCHIP (causal GP / model):",
              {k: (round(v["median_ratio_gp_causal_over_pchip"], 2), round(v["median_ratio_model_over_pchip"], 2))
               for k, v in bg.items()})
        E[f"by_gap_{tgt}"] = bg
    OUT["E"] = E


def main():
    part_a()
    part_b()
    part_c()
    part_d()
    part_e()
    Path("data").mkdir(exist_ok=True)
    Path("data/.foundations.json").write_text(json.dumps(OUT, indent=2, default=float), encoding="utf-8")
    print("")
    print("wrote data/.foundations.json")


if __name__ == "__main__":
    main()
