"""P05: the 3 keV CES_TI cut is a pedestal-pressure-limit constraint; spikes are fit failures.
Parts: A pressure limit, B power/time-scale, C Gaussian-fit contamination Monte Carlo,
D data checks on data/*.csv (641 shots). ASCII prints only. Writes out.json next to the script.
Run from repo root:  py <this file>            (optional:  py <this file> --skip-mc)
"""
import glob, json, os, sys, time
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import mannwhitneyu

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("CES_DATA_DIR", "data")
OUT = {}
e = 1.602176634e-19; mu0 = 4e-7 * np.pi
T0 = time.time()

# ----------------------------------------------------------------- A. pressure limit
Ip, BT, a, R, kap, q95 = 0.6e6, 1.8, 0.5, 1.8, 1.8, 4.0           # theory_plasma_physics.md App. A
Bpol = mu0 * Ip / (2 * np.pi * a * np.sqrt((1 + kap**2) / 2))
p_pol = Bpol**2 / (2 * mu0)                                       # beta_pol,ped = p_ped / p_pol
A = {"B_pol_T": Bpol, "p_pol_Pa": p_pol}
closure = {}
for alpha_c in (2.0, 3.0, 4.0, 6.0):
    for g in (1.0, 0.5):
        dpdr = alpha_c * BT**2 / (2 * mu0 * R * q95**2)
        K = dpdr * g * a * 0.076 * np.sqrt(2 * mu0) / Bpol
        p = K**2
        closure[f"alpha{alpha_c:g}_g{g:g}"] = {"dpdr_Pa_per_m": dpdr, "p_ped_max_Pa": p,
                                                "beta_pol_ped": p / p_pol,
                                                "Delta_r_m": g * a * 0.076 * np.sqrt(p / p_pol)}
A["closure"] = closure
for bN in (2.8, 3.5):
    betaT = bN * (Ip / 1e6) / (a * BT) / 100
    A[f"troyon_bN{bN:g}"] = {"beta_T": betaT, "p_avg_max_Pa": betaT * BT**2 / (2 * mu0)}
for bNp in (0.6, 1.0):
    betaTp = bNp * (Ip / 1e6) / (a * BT) / 100
    A[f"eped_bNped{bNp:g}"] = {"p_ped_Pa": betaTp * BT**2 / (2 * mu0),
                               "beta_pol_ped": betaTp * BT**2 / (2 * mu0) / p_pol}
Tmax = {}
p_lo = closure["alpha4_g1"]["p_ped_max_Pa"]; p_hi = closure["alpha6_g1"]["p_ped_max_Pa"]
for n in (2e19, 3e19, 5e19):
    for ratio_name, ratio in (("Ti=Te", 1.0), ("Ti=1.5Te", 1.5)):
        f = 1 + 1 / ratio
        Tmax[f"n{n:.0e}_{ratio_name}"] = {"Ti_max_lo_eV": p_lo / (n * e * f), "Ti_max_hi_eV": p_hi / (n * e * f),
                                           "beta_pol_needed_for_3keV": n * e * 3000 * f / p_pol,
                                           "n_max_for_3keV_at_p_hi": p_hi / (e * 3000 * f)}
A["Ti_ped_max"] = Tmax
OUT["A"] = A

# ----------------------------------------------------------------- D. data checks
files = sorted(glob.glob(os.path.join(DATA, "*.csv")))
if len(files) == 0:
    sys.exit("no CSVs")
GAP, CUT, ISO = 0.5, 3000.0, 2000.0
runs = []; shots = set(); n_obs = 0; n_obs_vt = 0
band = {"2500-3000": 0, "3000-4000": 0, ">4000": 0, ">2500": 0}
cols = ["ti", "ti_nm", "vt", "vt_nm", "vt_hf", "vt_nm_hf", "dE", "dB", "dM", "dE2", "sustained"]
acc = {c: [] for c in cols}
ti_values = []

def nbr_mean(t, v):
    idx = np.flatnonzero(~np.isnan(v)); out = np.full(v.shape[0], np.nan)
    if idx.size < 3: return out
    tt = t[idx]; vv = v[idx]
    okb = (np.diff(tt[:-1]) < GAP) & (np.diff(tt[1:]) < GAP)
    out[idx[1:-1][okb]] = 0.5 * (vv[:-2][okb] + vv[2:][okb])
    return out

for path in files:
    df = pd.read_csv(path)
    shot = os.path.basename(path)
    t = df["time"].to_numpy(float); ti = df["CES_TI"].to_numpy(float); vt = df["CES_VT"].to_numpy(float)
    ecei = df[[c for c in df.columns if c.startswith("ECEI_")]].to_numpy(float)
    bes = df[[c for c in df.columns if c.startswith("BES_")]].to_numpy(float)
    mc = df[[c for c in df.columns if c.startswith("MC")]].to_numpy(float)
    ok_step = np.concatenate(([False], np.abs(np.diff(t) - 0.01) < 1e-4))
    def adiff(x):
        d = np.full(x.shape[0], np.nan); d[1:] = np.mean(np.abs(np.diff(x, axis=0)), axis=1); d[~ok_step] = np.nan; return d
    dE, dB, dM = adiff(ecei), adiff(bes), adiff(mc)
    dE2 = dE.copy(); dE2[:-1] = np.fmax(dE[:-1], dE[1:])
    obs_vt_idx = np.flatnonzero(~np.isnan(vt)); vt_hf = vt.copy()
    if obs_vt_idx.size > 1:
        ov = vt[obs_vt_idx]; ot = t[obs_vt_idx]
        dup = (ov[1:] == ov[:-1]) & (np.diff(ot) < GAP)
        vt_hf[obs_vt_idx[1:][dup]] = np.nan
    vt_nm, vt_nm_hf, ti_nm = nbr_mean(t, vt), nbr_mean(t, vt_hf), nbr_mean(t, ti)
    idx = np.flatnonzero(~np.isnan(ti)); n_obs += idx.size
    n_obs_vt += int((~np.isnan(vt[idx])).sum())
    tv = ti[idx]; ti_values.append(tv)
    band["2500-3000"] += int(((tv > 2500) & (tv <= 3000)).sum()); band["3000-4000"] += int(((tv > 3000) & (tv <= 4000)).sum())
    band[">4000"] += int((tv > 4000).sum()); band[">2500"] += int((tv > 2500).sum())
    sustained = np.zeros(ti.shape[0], dtype=bool)
    if idx.size:
        cuts = np.flatnonzero(np.diff(t[idx]) >= GAP) + 1
        for seg_idx in np.split(idx, cuts):
            seg = ti[seg_idx]; n = seg.size; above = seg > CUT
            if not above.any(): continue
            d = np.diff(np.concatenate(([0], above.astype(int), [0])))
            for s0, s1 in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
                shots.add(shot)
                prev = seg[s0 - 1] if s0 > 0 else np.nan; nxt = seg[s1] if s1 < n else np.nan
                nb = [x for x in (prev, nxt) if not np.isnan(x)]
                iso = (s1 - s0 == 1) and len(nb) == 2 and all(x < ISO for x in nb)
                ratio = float(seg[s0:s1].max() / np.mean(nb)) if nb and np.mean(nb) > 0 else np.nan
                runs.append((int(s1 - s0), bool(iso), ratio, float(seg[s0:s1].max() - np.mean(nb)) if nb else np.nan))
                if s1 - s0 >= 5: sustained[seg_idx[s0:s1]] = True
    for c, arr_ in zip(cols, (ti, ti_nm, vt, vt_nm, vt_hf, vt_nm_hf, dE, dB, dM, dE2, sustained.astype(float))):
        acc[c].append(arr_[idx])
X = {c: np.concatenate(v) for c, v in acc.items()}
print(f"[census done in {time.time()-T0:.0f}s, {n_obs} TI rows]")

L = np.array([r[0] for r in runs]); I = np.array([r[1] for r in runs]); Rt = np.array([r[2] for r in runs]); Dm = np.array([r[3] for r in runs])
tiv = np.concatenate(ti_values)
is_sp = X["ti"] > CUT
D = {"n_files": len(files), "n_ti_observed": n_obs, "n_rows_gt_3keV": int(is_sp.sum()), "n_runs": int(L.size),
     "n_shots_with_runs": len(shots), "single_row_runs": int((L == 1).sum()), "single_row_share": float((L == 1).mean()),
     "runs_2_4": int(((L >= 2) & (L <= 4)).sum()), "runs_ge5": int((L >= 5).sum()), "max_run": int(L.max()),
     "isolated_share": float(I.mean()), "ratio_median": float(np.nanmedian(Rt)),
     "ratio_p25_p75": [float(np.nanpercentile(Rt, 25)), float(np.nanpercentile(Rt, 75))],
     "spike_minus_nbr_median_eV": float(np.nanmedian(Dm)), "spike_minus_nbr_p90_eV": float(np.nanpercentile(Dm, 90)),
     "spike_minus_nbr_max_eV": float(np.nanmax(Dm)),
     "band_counts": band, "ti_p99_eV": float(np.percentile(tiv, 99)), "ti_median_eV": float(np.median(tiv)),
     "ti_max_eV": float(tiv.max()),
     "nbr_mean_median_eV_at_spike_rows": float(np.nanmedian(X["ti_nm"][is_sp])),
     "spike_value_median_eV": float(np.median(X["ti"][is_sp])),
     "sustained_ge5_rows": int(X["sustained"].sum()),
     "sustained_ge5_value_median_eV": float(np.median(X["ti"][X["sustained"] > 0.5])) if X["sustained"].sum() else None}
# (ii) width-only prediction
ctrl = (~is_sp) & ~np.isnan(X["ti_nm"])
def mw(x, y):
    x = x[~np.isnan(x)]; y = y[~np.isnan(y)]
    return {"n_spike": int(x.size), "n_ctrl": int(y.size), "median_spike": float(np.median(x)), "median_ctrl": float(np.median(y)),
            "p90_spike": float(np.percentile(x, 90)), "p90_ctrl": float(np.percentile(y, 90)),
            "p_mannwhitney_two_sided": float(mannwhitneyu(x, y, alternative="two-sided").pvalue),
            "frac_spike_gt_100": float((x > 100).mean()), "frac_ctrl_gt_100": float((y > 100).mean()),
            "frac_spike_gt_300": float((x > 300).mean()), "frac_ctrl_gt_300": float((y > 300).mean())}
dvt = np.abs(X["vt"] - X["vt_nm"]); dvt_hf = np.abs(X["vt_hf"] - X["vt_nm_hf"])
vt_ratio = np.abs(X["vt_hf"] - X["vt_nm_hf"]) / np.maximum(np.abs(X["vt_nm_hf"]), 20.0)
ti_ratio = np.abs(X["ti"] - X["ti_nm"]) / X["ti_nm"]
D["ii"] = {"vt_observed_frac_at_spike_rows": float((~np.isnan(X["vt"][is_sp])).mean()),
           "vt_observed_frac_all_ti_rows": n_obs_vt / n_obs,
           "vt_observed_frac_ctrl_rows": float((~np.isnan(X["vt"][ctrl])).mean()),
           "vt_heldfree_observed_frac_at_spike_rows": float((~np.isnan(X["vt_hf"][is_sp])).mean()),
           "vt_heldfree_observed_frac_ctrl_rows": float((~np.isnan(X["vt_hf"][ctrl])).mean()),
           "abs_dVT_raw": mw(dvt[is_sp], dvt[ctrl]), "abs_dVT_heldfree": mw(dvt_hf[is_sp], dvt_hf[ctrl]),
           "same_row_relative_jump_median": {"ti": float(np.nanmedian(ti_ratio[is_sp])), "vt_heldfree": float(np.nanmedian(vt_ratio[is_sp])),
                                             "vt_heldfree_ctrl": float(np.nanmedian(vt_ratio[ctrl]))},
           "same_row_frac_vt_jump_ge_2x_level_heldfree": float(np.nanmean(vt_ratio[is_sp] >= 2.0)),
           "same_row_frac_vt_jump_ge_2x_level_heldfree_ctrl": float(np.nanmean(vt_ratio[ctrl] >= 2.0)),
           "same_row_n_vt_heldfree_with_nbrs": int((~np.isnan(vt_ratio[is_sp])).sum())}
# (iii) fast-diagnostic activity
def cmp(col):
    x = X[col][is_sp]; y = X[col][~is_sp]; x = x[~np.isnan(x)]; y = y[~np.isnan(y)]
    thr99 = np.percentile(y, 99)
    return {"median_spike": float(np.median(x)), "median_all": float(np.median(y)), "median_ratio": float(np.median(x) / np.median(y)),
            "p_mannwhitney": float(mannwhitneyu(x, y, alternative="two-sided").pvalue),
            "frac_spike_rows_above_all_p99": float((x > thr99).mean()), "n_spike": int(x.size), "n_all": int(y.size)}
D["iii"] = {"abs_dECEI": cmp("dE"), "abs_dBES": cmp("dB"), "abs_dMC": cmp("dM"), "abs_dECEI_two_sided": cmp("dE2")}
dEv = X["dE"]; okv = ~np.isnan(dEv)
qs = np.percentile(dEv[okv], [0, 10, 50, 90, 99, 100]); rate = {}
for lo, hi, nm in ((qs[0], qs[1], "bottom10"), (qs[1], qs[3], "mid10-90"), (qs[3], qs[4], "top10-1"), (qs[4], qs[5], "top1")):
    m = okv & (dEv >= lo) & (dEv <= hi); rate[nm] = {"n": int(m.sum()), "spike_rate": float(is_sp[m].mean())}
D["iii"]["spike_rate_by_dECEI_band"] = rate
# dips (<= 0.5 x nbr) as the other half of the single-sample outlier population, for contrast
dip = (~is_sp) & (X["ti"] <= 0.5 * X["ti_nm"])
D["iii"]["dip_rows_abs_dECEI_median_ratio"] = float(np.nanmedian(X["dE"][dip]) / np.nanmedian(X["dE"][~is_sp]))
OUT["D"] = D

# ----------------------------------------------------------------- B. power / time-scale
n_e = 3e19; dt_grid = 0.010; Delta_ped = 0.03
S = 4 * np.pi**2 * R * a * np.sqrt((1 + kap**2) / 2); V_ped = S * Delta_ped; V_tot = 2 * np.pi**2 * R * a**2 * kap
P_assumed_MW = 3.0; P_installed_MW = 10.0
B = {"S_m2": S, "V_ped_m3": V_ped, "V_tot_m3": V_tot, "V_ped_frac": V_ped / V_tot}
for nm, dT in (("median", D["spike_minus_nbr_median_eV"]), ("p90", D["spike_minus_nbr_p90_eV"]), ("max", D["spike_minus_nbr_max_eV"])):
    dW = 1.5 * n_e * dT * e * V_ped; P = dW / dt_grid
    B[nm] = {"dT_eV": dT, "dW_kJ": dW / 1e3, "P_MW_ions_pedestal_only": P / 1e6,
             "ratio_to_P_assumed_3MW": P / 1e6 / P_assumed_MW, "ratio_to_P_installed_10MW": P / 1e6 / P_installed_MW,
             "ratio_to_pedestal_share_of_ion_heating": P / 1e6 / (P_assumed_MW * 0.5 * V_ped / V_tot),
             "P_MW_if_Te_follows": 2 * P / 1e6}
Tsp = D["spike_value_median_eV"]
p_spike = n_e * e * (Tsp + D["nbr_mean_median_eV_at_spike_rows"])
B["p_at_median_spike_Pa"] = p_spike; B["p_spike_over_p_ped_max_hi"] = p_spike / p_hi
vA = BT / np.sqrt(mu0 * n_e * 2 * 1.6726e-27)
B["v_A_m_per_s"] = vA; B["mhd_growth_time_s"] = 1 / (0.1 * vA / R)
W_ped_i = 1.5 * n_e * D["nbr_mean_median_eV_at_spike_rows"] * e * V_ped
B["W_ped_ions_kJ_at_nbr_T"] = W_ped_i / 1e3; B["P_loss_normal_MW_tauE77ms"] = W_ped_i / 0.077 / 1e6
B["tau_eq_ms_doc_500eV_3e19"] = 13.4
OUT["B"] = B

# ----------------------------------------------------------------- C. Gaussian-fit contamination MC
if "--skip-mc" not in sys.argv:
    rng = np.random.default_rng(12345)
    NPIX, SIG, N_LINE, BG = 48, 4.0, 300.0, 2.0
    x = np.arange(NPIX, dtype=float); MU = 23.5
    def gauss(x, A, mu, sig, b): return A * np.exp(-0.5 * ((x - mu) / sig) ** 2) + b
    def spectrum(n_line=N_LINE, mu=MU, sig=SIG, bg=BG):
        lam = n_line / (sig * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sig) ** 2) + bg
        return rng.poisson(lam).astype(float)
    def fit(y, fix_bg=None):
        try:
            if fix_bg is None:
                p0 = [max(y.max() - np.median(y), 1), x[np.argmax(y)], 3.0, np.median(y)]
                popt, _ = curve_fit(gauss, x, y, p0=p0, bounds=([0, 0, 0.3, 0], [np.inf, NPIX, 60, np.inf]), maxfev=4000)
                return popt[1], popt[2]
            g3 = lambda x, A, mu, sig: gauss(x, A, mu, sig, fix_bg)
            p0 = [max(y.max() - fix_bg, 1), x[np.argmax(y)], 3.0]
            popt, _ = curve_fit(g3, x, y, p0=p0, bounds=([0, 0, 0.3], [np.inf, NPIX, 60]), maxfev=4000)
            return popt[1], popt[2]
        except Exception:
            return np.nan, np.nan
    VTH = 69.0
    def summarize(mus, sigs):
        mus = np.asarray(mus); sigs = np.asarray(sigs); ok = ~np.isnan(sigs)
        Tr = (sigs[ok] / SIG) ** 2; dv = (mus[ok] - MU) / SIG * VTH
        return {"n_ok": int(ok.sum()), "T_ratio_median": float(np.median(Tr)), "T_ratio_p90": float(np.percentile(Tr, 90)),
                "T_ratio_max": float(Tr.max()), "frac_T_ratio_gt_3": float((Tr > 3).mean()), "frac_T_ratio_gt_13": float((Tr > 13).mean()),
                "dV_median_kms": float(np.median(dv)), "dV_abs_median_kms": float(np.median(np.abs(dv))), "dV_abs_p90_kms": float(np.percentile(np.abs(dv), 90)),
                "T_rel_sd": float(np.std(np.sqrt(Tr)) * 2)}
    C = {}; NT = 300
    m, s = zip(*[fit(spectrum()) for _ in range(NT)]); C["0_clean"] = summarize(m, s); C["0_clean"]["CRB_rel_T"] = float(np.sqrt(2 / N_LINE))
    peak = N_LINE / (SIG * np.sqrt(2 * np.pi))
    for d in (0, 2, 4, 6, 10):
        for k in (3, 10):
            mm, ss = [], []
            for _ in range(NT):
                y = spectrum(); px = int(round(MU + d * SIG)); y[min(px, NPIX - 1)] += k * peak
                a_, b_ = fit(y); mm.append(a_); ss.append(b_)
            C[f"1_hotpixel_d{d}sig_k{k}"] = summarize(mm, ss)
    for nl in (30, 10, 3, 0):
        mm, ss = zip(*[fit(spectrum(n_line=nl)) for _ in range(NT)]); C[f"2_lowsignal_N{nl}"] = summarize(mm, ss)
    for d in (1, 2, 3, 4):
        mm, ss = [], []
        for _ in range(NT):
            y = spectrum(n_line=N_LINE / 2, mu=MU - d * SIG) + spectrum(n_line=N_LINE / 2, mu=MU + d * SIG, bg=0)
            a_, b_ = fit(y); mm.append(a_); ss.append(b_)
        C[f"3_twocomp_halfsep{d}sig"] = summarize(mm, ss); C[f"3_twocomp_halfsep{d}sig"]["predicted_T_ratio"] = 1 + d**2
    for step in (0.5, 1, 2, 5):
        mm, ss = zip(*[fit(spectrum(bg=BG + step * peak), fix_bg=BG) for _ in range(NT)]); C[f"4_bgstep_{step:g}peak_pinned"] = summarize(mm, ss)
    OUT["C"] = C
    print(f"[mc done in {time.time()-T0:.0f}s]")

with open(os.path.join(HERE, "out.json"), "w") as fh:
    json.dump(OUT, fh, indent=2, default=float)

def pr(k, v, ind=0):
    if isinstance(v, dict):
        print(" " * ind + str(k) + ":"); [pr(kk, vv, ind + 2) for kk, vv in v.items()]
    elif isinstance(v, float): print(" " * ind + f"{k}: {v:.4g}")
    else: print(" " * ind + f"{k}: {v}")
for k, v in OUT.items(): pr(k, v)
print("wrote", os.path.join(HERE, "out.json"))
