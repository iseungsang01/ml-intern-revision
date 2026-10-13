"""P09 check: dimensionless similarity MAST vs KSTAR, transferable-prior parameter count,
AR(1) sample-size formula at rho = 0.884, and the section-8az k-label curve.
No model is trained or scored. ASCII-only output (cp949 console).
Run from repo root:  py -I <this file>
"""
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(r"C:\Users\lss\Documents\GitHub\ml-intern-revision")
OUT = Path(__file__).resolve().parent / "p09_out.json"
res = {}

# ---------------------------------------------------------------------------
# 1. Dimensionless similarity (Kadomtsev 1975 / Connor-Taylor 1977): with n, T, B, L
#    there are exactly three independent dimensionless plasma parameters, rho*, nu*, beta,
#    plus the geometric/shape ones q, eps, kappa and the ratios T_e/T_i, M.
# ---------------------------------------------------------------------------
e = 1.602176634e-19
m_p = 1.67262192e-27
mu0 = 4e-7 * math.pi
lnL = 17.0

def dimensionless(name, T_eV, n_m3, B_T, R, a, q, kappa, M=2.0, Te_over_Ti=1.0):
    m_i = M * m_p
    T_J = T_eV * e
    vth = math.sqrt(2 * T_J / m_i)                 # m/s  (same convention as theory doc 1.1)
    Omega = e * B_T / m_i
    rho_i = vth / Omega
    eps = a / R
    rho_star = rho_i / a
    # Braginskii ion-ion collision frequency, NRL formulary: nu_ii = 4.80e-8 n_cm3 lnL mu^-1/2 T_eV^-3/2 s^-1
    nu_ii = 4.80e-8 * (n_m3 * 1e-6) * lnL / math.sqrt(M) * T_eV ** -1.5
    nu_star = nu_ii * q * R / (eps ** 1.5 * vth)   # theory doc 11.2 definition
    p = n_m3 * (T_J + T_J * Te_over_Ti)            # T_e = Te_over_Ti * T_i, both species
    beta = 2 * mu0 * p / B_T ** 2
    vA = B_T / math.sqrt(mu0 * n_m3 * m_i)
    chi_B = T_J / (16 * e * B_T)                   # m^2/s
    chi_gB = rho_star * chi_B
    f_trapped = math.sqrt(2 * eps / (1 + eps))     # fraction-of-velocity-space estimate, doc 1.3
    return dict(name=name, T_eV=T_eV, n_m3=n_m3, B_T=B_T, R=R, a=a, q=q, kappa=kappa, M=M,
                eps=eps, rho_i_mm=rho_i * 1e3, rho_star=rho_star, nu_ii_s=nu_ii, nu_star=nu_star,
                beta=beta, beta_pct=100 * beta, vA_km_s=vA / 1e3, chi_B=chi_B, chi_gB=chi_gB,
                trapped_fraction=min(f_trapped, 1.0), vth_km_s=vth / 1e3)

def ipb98(Ip_MA, B, P_MW, n19, M, R, eps, kappa):
    return 0.0562 * Ip_MA**0.93 * B**0.15 * P_MW**-0.69 * n19**0.41 * M**0.19 * R**1.97 * eps**0.58 * kappa**0.78

# KSTAR pedestal assumptions = docs/theory_plasma_physics.md 1.1 / 11.1 / 11.2 (T_i 600 eV, n 3e19, B 1.8 T, R 1.8, a 0.5, kappa 1.8)
# q = 4 at the pedestal reproduces the doc's nu* = 0.24 (checked below).
K = dimensionless("KSTAR", 600.0, 3e19, 1.8, 1.8, 0.5, 4.0, 1.8)
# MAST assumptions: pedestal T 200-300 eV, n 3-5e19 (task statement); geometry R 0.85 m, a 0.65 m
# (MAST-U published design figures; original MAST 0.9 / 0.6), B_T 0.5 T, kappa ~2, q_ped ~ 6 (assumption, ST q95 5-8).
M_lo = dimensionless("MAST_lo(T300,n3e19)", 300.0, 3e19, 0.5, 0.85, 0.65, 6.0, 2.0)
M_mid = dimensionless("MAST_mid(T250,n4e19)", 250.0, 4e19, 0.5, 0.85, 0.65, 6.0, 2.0)
M_hi = dimensionless("MAST_hi(T200,n5e19)", 200.0, 5e19, 0.5, 0.85, 0.65, 6.0, 2.0)

print("== 1. dimensionless table (pedestal, deuterium, T_e = T_i) ==")
hdr = ["name", "eps", "kappa", "q", "rho_i_mm", "rho_star", "nu_star", "beta_pct", "trapped_fraction", "chi_gB"]
print(" | ".join(hdr))
for d in (K, M_lo, M_mid, M_hi):
    print(" | ".join(f"{d[h]:.4g}" if isinstance(d[h], float) else str(d[h]) for h in hdr))
print("check vs theory doc 11.2: KSTAR rho*=%.4f (doc 0.0055)  nu*=%.3f (doc 0.24)  beta=%.2f%% (doc 0.45%%)  rho_i=%.2f mm (doc 2.8)"
      % (K["rho_star"], K["nu_star"], K["beta_pct"], K["rho_i_mm"]))
ratios = {k: M_mid[k] / K[k] for k in ("rho_star", "nu_star", "beta", "eps", "kappa", "q")}
ratio_range = {k: (min(M_lo[k], M_mid[k], M_hi[k]) / K[k], max(M_lo[k], M_mid[k], M_hi[k]) / K[k])
               for k in ("rho_star", "nu_star", "beta")}
print("MAST/KSTAR ratios (mid):", {k: round(v, 3) for k, v in ratios.items()})
print("MAST/KSTAR ratio range over T 200-300 eV, n 3-5e19:", {k: (round(v[0], 3), round(v[1], 3)) for k, v in ratio_range.items()})
# IPB98 confinement times and the implied 10 ms AR(1) coefficient
tauE_K = ipb98(0.6, 1.8, 3.0, 3.0, 2, 1.8, K["eps"], 1.8)
tauE_M = ipb98(0.7, 0.5, 2.0, 4.0, 2, 0.85, M_mid["eps"], 2.0)
rho10_K = math.exp(-0.010 / tauE_K)
rho10_M = math.exp(-0.010 / tauE_M)
print("IPB98 tau_E: KSTAR %.1f ms (doc 77), MAST %.1f ms; implied 10 ms AR(1) rho: %.3f vs %.3f" % (tauE_K * 1e3, tauE_M * 1e3, rho10_K, rho10_M))
res["dimensionless"] = dict(KSTAR=K, MAST_lo=M_lo, MAST_mid=M_mid, MAST_hi=M_hi, ratios_mid=ratios, ratio_range=ratio_range,
                            tauE_ms=dict(KSTAR=tauE_K * 1e3, MAST=tauE_M * 1e3), rho_10ms=dict(KSTAR=rho10_K, MAST=rho10_M))

# ---------------------------------------------------------------------------
# 2. Parameter accounting of the TokaMind arm (b12_models.py constants) vs an OU+impulse generator
# ---------------------------------------------------------------------------
D, FF, L = 192, 768, 4
per_layer = (3 * D * D + 3 * D) + (D * D + D) + (D * FF + FF) + (FF * D + D) + 2 * (2 * D)
backbone = L * per_layer
emb_pre = (50 + 1) * D + 3 * D + 3 * D                # pos, mod, role (transferred)
groups = [9, 4, 2, 9, 9]                             # BES, ECEI, MC, dt+8 Ti slow, dt+8 Vt slow
proj_new = sum((g + 1) * D for g in groups)
group_embed_new = 5 * D
heads_new = 2 * ((D * 64 + 64) + (64 * 1 + 1))
total = backbone + emb_pre + proj_new + group_embed_new + heads_new
pre = backbone + emb_pre
print("== 2. parameter accounting ==")
print("backbone %d + transferred embeddings %d = pretrained %d; new proj %d + group_embed %d + heads %d; total %d (8az: 1,823,490)"
      % (backbone, emb_pre, pre, proj_new, group_embed_new, heads_new, total))
print("pretrained fraction %.4f; KSTAR-instrument-specific projections fraction %.4f" % (pre / total, proj_new / total))
ou_params = {"OU": ["sigma_s", "tau", "sigma_n"], "OU+level": ["mu"], "OU+impulse": ["lambda_impulse", "A_impulse"]}
n_ou = sum(len(v) for v in ou_params.values())
print("OU generator: 3 (sigma_s, tau, sigma_n) + 1 level + 2 impulse = %d parameters; ratio weights/params = %.2e" % (n_ou, total / n_ou))
res["params"] = dict(backbone=backbone, emb_pre=emb_pre, pretrained=pre, proj_new=proj_new, group_embed_new=group_embed_new,
                     heads_new=heads_new, total=total, matches_8az=(total == 1823490), pre_frac=pre / total, proj_frac=proj_new / total,
                     ou_params=ou_params, n_ou=n_ou)
assert total == 1823490, total

# ---------------------------------------------------------------------------
# 3. AR(1) sample size at rho = 0.884 (data/.foundations.json D.CES_TI.fits.robust.pred.rho_1step)
# ---------------------------------------------------------------------------
fnd = json.load(open(REPO / "data" / ".foundations.json"))
fit = fnd["D"]["CES_TI"]["fits"]["robust"]
rho = fit["pred"]["rho_1step"]
tau_ms, sig_s, sig_n = fit["tau_ms"], fit["sigma_s"], fit["sigma_n"]
dt = 10.0
print("== 3. AR(1) sample size ==")
print("rho_1step = %.4f  (tau = %.1f ms, sigma_s = %.1f eV, sigma_n = %.1f eV)" % (rho, tau_ms, sig_s, sig_n))
# SE(rho_hat) ~ sqrt((1 - rho^2) / n)  (Bartlett / asymptotic OLS variance of the AR(1) coefficient)
def n_for_rel(rel):
    return (1 - rho**2) / (rel * rho) ** 2
n_rho10 = n_for_rel(0.10)
# tau = -dt / ln(rho): d tau / d rho = dt / (rho ln^2 rho)  -> rel SE(tau) = SE(rho) / (rho |ln rho|)
g = rho * abs(math.log(rho))
n_tau10 = (1 - rho**2) / (0.10 * g) ** 2
# variance sigma^2: rel SE = sqrt(2 / n_eff), n_eff = n (1 - rho) / (1 + rho)
n_var10 = (2 / 0.10**2) * (1 + rho) / (1 - rho)
print("rows for 10%% on rho: %.0f ; on tau: %.0f ; on sigma_s^2: %.0f" % (n_rho10, n_tau10, n_var10))
# Monte Carlo of the SE formula
rng = np.random.default_rng(0)
mc = {}
for n in (30, 300, 1800, 3000):
    est = []
    for _ in range(2000):
        x = np.empty(n); x[0] = rng.normal(0, 1)
        innov = rng.normal(0, math.sqrt(1 - rho**2), n)
        for t in range(1, n):
            x[t] = rho * x[t - 1] + innov[t]
        est.append(np.sum(x[1:] * x[:-1]) / np.sum(x[:-1] ** 2))
    est = np.array(est)
    mc[n] = dict(mean=float(est.mean()), sd=float(est.std()), formula=math.sqrt((1 - rho**2) / n))
    print("  n=%5d  MC sd(rho_hat)=%.4f  formula=%.4f  bias=%+.4f" % (n, mc[n]["sd"], mc[n]["formula"], mc[n]["mean"] - rho))
# how many rows does one KSTAR shot give? count observed CES_TI / CES_VT rows per file
data_dir = Path(os.environ.get("CES_DATA_DIR", REPO / "data"))
csvs = sorted(data_dir.glob("*.csv"))
if not csvs:
    sys.exit("no shot CSVs found in %s -- set CES_DATA_DIR (no synthetic data allowed)" % data_dir)
import pandas as pd
ti_rows, vt_rows, tot_rows = [], [], []
for f in csvs:
    df = pd.read_csv(f, usecols=["CES_TI", "CES_VT"])
    ti_rows.append(int(df["CES_TI"].notna().sum())); vt_rows.append(int(df["CES_VT"].notna().sum())); tot_rows.append(len(df))
ti_rows, vt_rows = np.array(ti_rows), np.array(vt_rows)
print("files %d; observed CES_TI rows per file: median %d, mean %.0f, total %d ; CES_VT: median %d, total %d"
      % (len(csvs), np.median(ti_rows), ti_rows.mean(), ti_rows.sum(), np.median(vt_rows), vt_rows.sum()))
shots_rho = n_rho10 / np.median(ti_rows); shots_tau = n_tau10 / np.median(ti_rows); shots_var = n_var10 / np.median(ti_rows)
print("median shots needed: rho %.2f, tau %.1f, sigma_s^2 %.1f ; SE(rho) from all %d rows = %.5f"
      % (shots_rho, shots_tau, shots_var, ti_rows.sum(), math.sqrt((1 - rho**2) / ti_rows.sum())))
bias_mast_prior = rho10_K - rho10_M
print("a MAST-born prior on rho (IPB98 ratio) would be off by %.3f = %.0f SE of the KSTAR estimate" % (bias_mast_prior, bias_mast_prior / math.sqrt((1 - rho**2) / ti_rows.sum())))
res["ar1"] = dict(rho=rho, tau_ms=tau_ms, sigma_s=sig_s, sigma_n=sig_n, n_rho10=n_rho10, n_tau10=n_tau10, n_var10=n_var10, mc=mc,
                  files=len(csvs), ti_rows_median=float(np.median(ti_rows)), ti_rows_total=int(ti_rows.sum()),
                  vt_rows_median=float(np.median(vt_rows)), vt_rows_total=int(vt_rows.sum()),
                  shots_rho=shots_rho, shots_tau=shots_tau, shots_var=shots_var,
                  se_rho_all=math.sqrt((1 - rho**2) / ti_rows.sum()), mast_prior_bias=bias_mast_prior)

# ---------------------------------------------------------------------------
# 4. Section 8az label-sparsified curve (cut population), skill vs PCHIP, from THESIS_RESULTS.md 8az table
# ---------------------------------------------------------------------------
k_curve = {"bilstm": {0: -0.025, 3: 0.146, 10: 0.407}, "np": {0: -0.040, 3: 0.148, 10: 0.403},
           "tm_pre": {0: -0.016, 3: 0.132, 10: 0.392}, "tm_scratch": {0: -0.003, 3: 0.137, 10: 0.398}, "GP": {0: 0.0, 3: 0.127, 10: 0.079}}
print("== 4. 8az k-label curve ==")
lift03 = {a: round(v[3] - v[0], 3) for a, v in k_curve.items() if a != "GP"}
lift310 = {a: round(v[10] - v[3], 3) for a, v in k_curve.items() if a != "GP"}
print("lift 0->3 :", lift03, " spread %.3f" % (max(lift03.values()) - min(lift03.values())))
print("lift 3->10:", lift310, " spread %.3f" % (max(lift310.values()) - min(lift310.values())))
pre_vs_scratch = {k: round(k_curve["tm_pre"][k] - k_curve["tm_scratch"][k], 3) for k in (0, 3, 10)}
print("tm_pre - tm_scratch at k=0,3,10:", pre_vs_scratch)
axis1 = {"cut_Ti": (-0.094, -0.090), "incl_Ti": (-0.039, -0.032), "cut_Vrot": (0.012, 0.021), "incl_Vrot": (-0.016, -0.003)}
axis1_diff = {k: round(abs(a - b), 3) for k, (a, b) in axis1.items()}
print("axis-1 |delta_pre - delta_scratch|:", axis1_diff, " (8az: 0.004/0.007 Ti, 0.009/0.013 Vrot)")
tol = 0.013
ok = all(abs(v) <= tol for v in pre_vs_scratch.values()) and max(lift310.values()) - min(lift310.values()) <= tol
print("all family differences within +/-%.3f : %s" % (tol, ok))
res["k_curve"] = dict(curve=k_curve, lift03=lift03, lift310=lift310, pre_vs_scratch=pre_vs_scratch, axis1_diff=axis1_diff, within_tol=ok)

OUT.write_text(json.dumps(res, indent=1, default=float))
print("wrote", OUT)

# ---------------------------------------------------------------------------
# 5. Time-domain form: SE(rho) = sqrt((1 - rho^2) Delta / T), record length T = n Delta
#    and the impulse-recovery kernel scales (theory doc 6.3)
# ---------------------------------------------------------------------------
print("== 5. time-domain sample size as record length T (Delta = 10 ms) ==")
Delta = 0.010
T_rho10, T_tau10, T_var10 = n_rho10 * Delta, n_tau10 * Delta, n_var10 * Delta
print("T for 10%%: rho %.2f s, tau %.1f s, sigma_s^2 %.1f s" % (T_rho10, T_tau10, T_var10))
# record length of a shot: observed-row span of the CES_TI label, from the time column
spans, labelled_s = [], []
for f in csvs:
    df = pd.read_csv(f, usecols=["time", "CES_TI"])
    obs = df.loc[df["CES_TI"].notna(), "time"]
    if len(obs) > 1:
        spans.append(float(obs.iloc[-1] - obs.iloc[0])); labelled_s.append(len(obs) * Delta)
spans = np.array(spans); labelled_s = np.array(labelled_s)
print("CES_TI labelled span per shot: median %.2f s (mean %.2f, p10 %.2f, p90 %.2f); labelled rows x Delta: median %.2f s; total labelled %.0f s"
      % (np.median(spans), spans.mean(), np.percentile(spans, 10), np.percentile(spans, 90), np.median(labelled_s), labelled_s.sum()))
print("rho_hat precision after one median shot: SE = %.4f (%.1f%% of rho); after the full record: %.5f"
      % (math.sqrt((1 - rho**2) * Delta / np.median(labelled_s)), 100 * math.sqrt((1 - rho**2) * Delta / np.median(labelled_s)) / rho,
         math.sqrt((1 - rho**2) * Delta / labelled_s.sum())))
# impulse kernel: ELM crash then recovery with tau_build 10-50 ms (doc 6.3); fraction of the 10 ms grid it spans
for tb in (10, 30, 50):
    print("  recovery kernel tau_build=%d ms: e-fold in %.1f grid steps; 95%% recovered after %.0f ms" % (tb, tb / 10, 3 * tb))
res["time_domain"] = dict(Delta_s=Delta, T_rho10_s=T_rho10, T_tau10_s=T_tau10, T_var10_s=T_var10,
                          span_median_s=float(np.median(spans)), span_p10=float(np.percentile(spans, 10)), span_p90=float(np.percentile(spans, 90)),
                          labelled_median_s=float(np.median(labelled_s)), labelled_total_s=float(labelled_s.sum()))
OUT.write_text(json.dumps(res, indent=1, default=float))
print("wrote", OUT)
