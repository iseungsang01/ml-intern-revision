"""P06 time-domain companion: information per sampling step Delta versus tau_eq.
Discretised driver model  y_t = rho y_{t-1} + (1-rho) zbar_t + eps_t,  rho = exp(-Delta/tau_eq),
Var eps = sigma_s^2 (1-rho^2).  Uses the OU fits in data/.foundations.json (classical and robust).
ASCII only."""
import json
import numpy as np
from pathlib import Path

ROOT = Path(r"C:\Users\lss\Documents\GitHub\ml-intern-revision")
F = json.load(open(ROOT / "data" / ".foundations.json", encoding="utf-8"))
R = json.load(open(Path(__file__).resolve().parent / "p06_results.json", encoding="utf-8"))
R2_MEAS = R["deep_history_h5"]["ch15+inc15"]["r2_partial_oos"]       # 0.066 (t, t-1 fast rows, 50 ms history)
R2_MEAS50 = R["deep_history_h5"]["ch15 lags0-5(75)"]["r2_partial_oos"]
DELTA = 10.0
print("empirical partial R2 given 50 ms history: %.4f (t,t-1) / %.4f (50 ms fast context)" % (R2_MEAS, R2_MEAS50))

for fit in ("classical", "robust"):
    p = F["D"]["CES_TI"]["fits"][fit]; pr = p["pred"]
    s_n2, s_s2 = p["sigma_n"] ** 2, p["sigma_s"] ** 2
    var_nu = pr["mse_filter_inf"]                     # P^- + sigma_n^2
    P_minus = var_nu - s_n2
    K = pr["kalman_gain"]; P = (1 - K) * P_minus
    rho_ou = pr["rho_1step"]
    share_eps = s_s2 * (1 - rho_ou ** 2) / var_nu
    print("\n[%s] sigma_n=%.1f sigma_s=%.1f  Var(nu)=%.0f  P^-=%.0f  P=%.0f  share of process noise in nu = %.3f, prior-error share = %.3f, meas-noise share = %.3f"
          % (fit, p["sigma_n"], p["sigma_s"], var_nu, P_minus, P, share_eps, rho_ou ** 2 * P / var_nu, s_n2 / var_nu))
    # (A) driver route: required per-step driver innovation sd to reproduce R2_MEAS, per tau_eq
    print("  (A) driver route: y_t = rho y_{t-1} + (1-rho) zbar_t + eps_t ; required sd of (zbar_t - E[zbar_t|h]) in eV for R2=%.3f" % R2_MEAS)
    print("      tau_eq[ms]   rho    (1-rho)/(1+rho)  Delta/(2tau)   I_step/nat at S_z=1  required sd_zbar|h [eV]  info rate I/Delta at S_z=1 [nat/s]")
    for tau in (8.0, 20.0, 30.0, 59.0, 81.2):
        rho = np.exp(-DELTA / tau)
        ratio = (1 - rho) / (1 + rho)
        I_step = 0.5 * np.log(1 + ratio * 1.0)             # S_z = Var(zbar|h)/sigma_s^2 = 1
        req = np.sqrt(R2_MEAS * var_nu) / (1 - rho)
        print("      %6.1f     %.3f     %.4f          %.4f        %.4f            %6.0f                 %.2f"
              % (tau, rho, ratio, DELTA / (2 * tau), I_step, req, I_step / (DELTA / 1000.0)))
    # (B) de-noising route with the S8ar level correlation r_L = 0.34 (z = a y + w, Gaussian, stationary)
    for r_L in (0.341, 0.311):
        q = (1.0 / r_L ** 2 - 1.0)                          # sigma_w^2 / (a^2 sigma_y^2)
        for lab, Pp in (("steady filter", P), ("one obs", s_s2 * s_n2 / (s_s2 + s_n2))):
            r2_e = Pp / (Pp + q * s_s2)                     # r^2(prior error, z | h)
            var_nu_lab = var_nu if lab == "steady filter" else pr["mse_filter_K"]["1"]
            contrib = rho_ou ** 2 * Pp * r2_e / var_nu_lab
            print("  (B) de-noising route, r_L=%.3f, history=%-13s: r2(prior err, z|h)=%.4f -> contribution to partial R2 of nu = %.4f" % (r_L, lab, r2_e, contrib))
    # (C) combined: driver route with S8ar rate correlation on the increment (r=0.07) mapped to eps
    var_inc = 2 * s_n2 + 2 * s_s2 * (1 - rho_ou)
    for r_rate in (0.070, 0.078):
        expl = r_rate ** 2 * var_inc                        # eV^2 explained in the increment
        print("  (C) rate route via S8ar r=%.3f: explains %.0f eV^2 of the increment (var %.0f) -> share of nu %.4f" % (r_rate, expl, var_inc, expl / var_nu))
