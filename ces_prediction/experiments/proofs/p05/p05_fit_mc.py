"""P05 part C: single-frame Gaussian-fit contamination Monte Carlo (ASCII prints only).
48-pixel spectrum, true sigma = 4 px, N_line = 300 photons (theory_foundations 1.3/1.5), background 2 ph/px.
Fit = 4-parameter Gaussian (A, mu, sigma, b) by least squares, as a generic CES pipeline would.
Cases: 0 clean; 1 hot pixel / cosmic ray (one pixel, k x peak, at offset d sigma); 2 low-signal frame
(beam notch / shutter, N_line small); 3 two components at separation 2d sigma; 4 background step not fitted.
Writes mc.json next to the script.  Run:  py -u <this file>
"""
import json, os, time, warnings
import numpy as np
from scipy.optimize import curve_fit, OptimizeWarning
warnings.simplefilter("ignore", OptimizeWarning); warnings.simplefilter("ignore", RuntimeWarning)
HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(12345)
NPIX, SIG, N_LINE, BG, MU, NT = 48, 4.0, 300.0, 2.0, 23.5, 300
x = np.arange(NPIX, dtype=float)
VTH = 69.0  # km/s per 1 sigma_px at 593 eV (theory_foundations 1.1)
def gauss(x, A, mu, sig, b): return A * np.exp(-0.5 * ((x - mu) / sig) ** 2) + b
def spectrum(n_line=N_LINE, mu=MU, sig=SIG, bg=BG):
    lam = n_line / (sig * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sig) ** 2) + bg
    return rng.poisson(lam).astype(float)
def fit(y, fix_bg=None):
    try:
        if fix_bg is None:
            p0 = [max(y.max() - np.median(y), 1.0), float(x[np.argmax(y)]), 3.0, float(np.median(y))]
            popt, _ = curve_fit(gauss, x, y, p0=p0, bounds=([0, 0, 0.3, 0], [np.inf, NPIX, 60, np.inf]), max_nfev=400)
            return popt[1], popt[2]
        g3 = lambda x, A, mu, sig: gauss(x, A, mu, sig, fix_bg)
        p0 = [max(y.max() - fix_bg, 1.0), float(x[np.argmax(y)]), 3.0]
        popt, _ = curve_fit(g3, x, y, p0=p0, bounds=([0, 0, 0.3], [np.inf, NPIX, 60]), max_nfev=400)
        return popt[1], popt[2]
    except Exception:
        return np.nan, np.nan
def summarize(mus, sigs):
    mus = np.asarray(mus); sigs = np.asarray(sigs); ok = ~np.isnan(sigs)
    Tr = (sigs[ok] / SIG) ** 2; dv = (mus[ok] - MU) / SIG * VTH
    return {"n_ok": int(ok.sum()), "T_ratio_median": float(np.median(Tr)), "T_ratio_p90": float(np.percentile(Tr, 90)),
            "T_ratio_max": float(Tr.max()), "frac_T_ratio_gt_3": float((Tr > 3).mean()), "frac_T_ratio_gt_13": float((Tr > 13).mean()),
            "dV_median_kms": float(np.median(dv)), "dV_abs_median_kms": float(np.median(np.abs(dv))), "dV_abs_p90_kms": float(np.percentile(np.abs(dv), 90)),
            "T_rel_sd": float(np.std(np.sqrt(Tr)) * 2)}
C = {}; t0 = time.time()
def run(name, gen, fix_bg=None):
    mm, ss = [], []
    for _ in range(NT):
        a_, b_ = fit(gen(), fix_bg); mm.append(a_); ss.append(b_)
    C[name] = summarize(mm, ss); print(f"{name}: T_ratio median {C[name]['T_ratio_median']:.3g} p90 {C[name]['T_ratio_p90']:.3g} "
                                      f"|dV| median {C[name]['dV_abs_median_kms']:.3g} km/s  ({time.time()-t0:.0f}s)", flush=True)
run("0_clean", spectrum); C["0_clean"]["CRB_rel_T"] = float(np.sqrt(2 / N_LINE))
peak = N_LINE / (SIG * np.sqrt(2 * np.pi))
for d in (0, 2, 4, 6, 10):
    for k in (3, 10):
        def gen(d=d, k=k):
            y = spectrum(); px = int(round(MU + d * SIG)); y[min(px, NPIX - 1)] += k * peak; return y
        run(f"1_hotpixel_d{d}sig_k{k}", gen)
for nl in (30, 10, 3, 0):
    run(f"2_lowsignal_N{nl}", lambda nl=nl: spectrum(n_line=nl))
for d in (1, 2, 3, 4):
    run(f"3_twocomp_halfsep{d}sig", lambda d=d: spectrum(n_line=N_LINE / 2, mu=MU - d * SIG) + spectrum(n_line=N_LINE / 2, mu=MU + d * SIG, bg=0))
    C[f"3_twocomp_halfsep{d}sig"]["predicted_T_ratio"] = 1 + d**2
for step in (0.5, 1, 2, 5):
    run(f"4_bgstep_{step:g}peak_pinned", lambda step=step: spectrum(bg=BG + step * peak), fix_bg=BG)
with open(os.path.join(HERE, "mc.json"), "w") as fh:
    json.dump(C, fh, indent=2)
print("wrote mc.json")
