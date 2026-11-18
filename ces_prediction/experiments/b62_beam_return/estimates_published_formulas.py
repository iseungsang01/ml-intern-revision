"""Estimates from published formulas, not measurements (supplement of the PPCF manuscript, S10 and S12).

(a) Beam-ion slowing down at KSTAR edge parameters and a zero-dimensional energy balance through a beam
    interruption: critical energy, Spitzer slowing-down time on electrons and Stix thermalisation time from
    Heidbrink & Sadler, Nucl. Fusion 34 (1994) 535, eqs. (7), (9), (10); a classical slowing-down kernel of a
    95 keV deuteron; dW/dt = P_h(t) - W/tau_E with P_h the beam source passed through that kernel.
    Reproduces the arithmetic of the literature notes (scratchpad beam/balance.py of 2026-11-18): same formulas,
    same inputs.
(b) Optical depth of perpendicular second-harmonic X-mode ECE at the KSTAR edge, from the absorption formulas
    (Hutchinson 5.2.39, 5.2.47, 5.2.54, relativistic line shape) as implemented in the KSTAR synthetic-diagnostic
    code syndia (eceint.py) and summarised by Zhu et al., IEEE Trans. Plasma Sci. 47 (2019) 2110, eq. (9.3).
    Reproduces data/literature/code_own_tau_x2_estimate.py.txt.

The plasma parameters are assumed (typical published KSTAR edge values), not measured for the discharges of the
data set. Writes docs/paper/b62_estimates.md. Run from the repository root:
    py ces_prediction/experiments/b62_beam_return/estimates_published_formulas.py
"""
import math
from pathlib import Path

import numpy as np
from scipy import integrate

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "docs" / "paper" / "b62_estimates.md"


# ---------------------------------------------------------------- (a) slowing down and energy balance
def params(te_ev, ne, zeff=2.0, lnl=17.0, a_beam=2.0):
    """Critical energy (keV) and slowing-down time on electrons (s); deuterium plasma with carbon."""
    n_d = (6 - zeff) / 5
    n_c = (zeff - 1) / 30
    z2a = n_d * 1 / 2 + n_c * 36 / 12               # <Z^2/A>
    ec = 14.8 * a_beam * te_ev * z2a ** (2 / 3) / 1e3
    tse = 6.3e14 * a_beam * te_ev ** 1.5 / (ne * lnl)
    return ec, tse


def kernel(e0, ec, tse, ti, dt=1e-5, tmax=1.0):
    """Power of one slowing ion to electrons and ions per step until it reaches 1.5 T_i."""
    e, t, pe, pi = e0, 0.0, [], []
    while e > 1.5 * ti and t < tmax:
        dee = 2 * e / tse
        dei = 2 * e / tse * (ec / e) ** 1.5
        pe.append(dee)
        pi.append(dei)
        e -= (dee + dei) * dt
        t += dt
    return np.array(pe), np.array(pi)


def response(te, ne, ti, notch, tau_e, f_off, e0=95.0, dt=1e-4, tmax=0.6):
    t = np.arange(-0.4, tmax, dt)
    s = np.ones_like(t)
    s[(t >= 0) & (t < notch)] = 1 - f_off
    ec, tse = params(te, ne)
    pe, pi = kernel(e0, ec, tse, ti, dt=dt)
    k = pe + pi
    k = k / k.sum()
    ph = np.convolve(s, k)[:len(t)]
    ph = ph / ph[(t > -0.05) & (t < 0)].mean()
    ph[t < -0.4 + len(k) * dt] = 1
    dp = ph - 1
    dw = np.zeros_like(t)
    for i in range(1, len(t)):
        dw[i] = dw[i - 1] + dt * (dp[i - 1] - dw[i - 1] / tau_e)
    x = dw / tau_e
    first = x[(t >= notch) & (t < notch + 0.01)].mean()
    during = x[(t >= 0) & (t < notch)].mean()
    i = int(np.argmin(x))
    return -during, -first, -x[i], t[i] - notch


# ---------------------------------------------------------------- (b) optical depth of X2 ECE
E = 1.602e-19
ME = 9.109e-31
EPS = 8.854e-12
C = 299792458.0
MC2 = ME * C ** 2


def tau_x2(ne, te_kev, r_res, b0=1.8, r0=1.8, m=2):
    te = te_kev * 1e3 * E
    br = b0 * r0
    omega = m * E * (br / r_res) / ME
    zeta = MC2 / te / 2
    wpe2 = ne * E ** 2 / (EPS * ME)
    rs = np.linspace(r_res - 0.06, r_res + 0.02, 4001)
    al = np.zeros_like(rs)
    for i, r in enumerate(rs):
        wce = E * (br / r) / ME
        mu = (omega / (m * wce)) ** 2
        x = wpe2 / (m * wce) ** 2
        n2 = 1 - x * (1 - x) / (1 - x - 1 / (2 * m ** 2) - np.sqrt((1 / (2 * m ** 2)) ** 2))
        n = np.sqrt(n2).real
        am = E ** 2 * ne / (4 * C * ME * EPS) * m ** (2 * m - 1) / math.factorial(m - 1) * (te / (2 * MC2)) ** (m - 1)
        eta = n ** (2 * m - 3) * (m - 1) ** 2 * (1 - (m + 1) / m / x * (1 - n ** 2)) ** 2
        if mu < 1:
            b1, b2 = -np.sqrt(1 - mu), np.sqrt(1 - mu)
            sh = 2 * np.pi * zeta ** 3.5 * omega / (np.sqrt(np.pi) * (m * wce) ** 2) * integrate.quad(
                lambda b: (1 - b ** 2 - mu) ** 2 * np.exp(-zeta * (1 - mu)), b1, b2)[0]
            al[i] = am * eta * sh
    return float(np.trapezoid(al, rs))


def main():
    lines = ["# Estimates from published formulas, not measurements",
             "",
             "Script: `ces_prediction/experiments/b62_beam_return/estimates_published_formulas.py`. The plasma "
             "parameters are assumed typical KSTAR edge values, not measurements of the discharges of the data set.",
             "",
             "## (a) Beam-ion slowing down (Heidbrink & Sadler 1994, eqs. 7, 9, 10)",
             "",
             "Inputs: deuterium beam, Z_eff = 2 (carbon), ln Lambda = 17, beam components 95, 47.5 and 31.7 keV.",
             "",
             "| location | T_e (eV) | n_e (m^-3) | E0 (keV) | E_crit (keV) | tau_se (ms) | tau_th (ms) | share to ions |",
             "|---|---|---|---|---|---|---|---|"]
    cases = {"pedestal top": (500, 2.5e19, 0.5), "hot pedestal top": (800, 3e19, 0.8),
             "mid-radius": (1500, 3.5e19, 1.5), "core": (2500, 4e19, 2.5)}
    print("(a) slowing down")
    for name, (te, ne, ti) in cases.items():
        ec, tse = params(te, ne)
        for e0 in (95.0, 95.0 / 2, 95.0 / 3):
            pe, pi = kernel(e0, ec, tse, ti)
            share = pi.sum() / (pe.sum() + pi.sum())
            tth = tse / 3 * np.log(1 + (e0 / ec) ** 1.5)
            row = (f"| {name} | {te} | {ne:.1e} | {e0:.1f} | {ec:.1f} | {tse * 1e3:.0f} | {tth * 1e3:.0f} | "
                   f"{share:.2f} |")
            lines.append(row)
            print(row)
    lines += ["",
              "## (a) Zero-dimensional balance through an interruption",
              "",
              "dW/dt = P_h(t) - W/tau_E; P_h = beam source (fraction f_off switched off for the interruption) "
              "passed through the slowing-down kernel of a 95 keV deuteron at the parameters of the row; "
              "self-similar profiles, so dT/T = dW/W at fixed density; ohmic and other heating neglected "
              "(upper estimates). Falls are positive numbers, in per cent of the level before.",
              "",
              "| kernel | interruption (ms) | tau_E (ms) | f_off | mean during (%) | mean over the first frame "
              "after return (%) | largest (%) | time of largest after return (ms) |",
              "|---|---|---|---|---|---|---|---|"]
    print("(a) balance")
    for name in ("pedestal top", "mid-radius", "core"):
        te, ne, ti = cases[name]
        for notch in (0.0105, 0.027):
            for tau_e in (0.05, 0.085):
                for f_off, flab in ((1.0, "1"), (1 / 3, "1/3")):
                    d, f1, mx, tm = response(te, ne, ti, notch, tau_e, f_off)
                    row = (f"| {name} | {notch * 1e3:.1f} | {tau_e * 1e3:.0f} | {flab} | {d * 100:.1f} | "
                           f"{f1 * 100:.1f} | {mx * 100:.1f} | {tm * 1e3:.0f} |")
                    lines.append(row)
                    print(row)
    lines += ["",
              "## (b) Optical depth of perpendicular X2 ECE (B_T = 1.8 T at R0 = 1.8 m)",
              "",
              "Absorption of Hutchinson eqs. 5.2.39, 5.2.47, 5.2.54 with the relativistic line shape, as in "
              "syndia eceint.py (see also Zhu et al. 2019, eq. 9.3), integrated across the resonance of a "
              "uniform slab.",
              "",
              "| case | n_e (m^-3) | T_e (keV) | R (m) | tau |",
              "|---|---|---|---|---|"]
    print("(b) optical depth")
    for ne, te, r, lab in ((2e19, 0.5, 2.198, "pedestal top, dense"), (1.5e19, 0.3, 2.198, "pedestal top"),
                           (1e19, 0.2, 2.22, "low density"), (0.8e19, 0.1, 2.25, "separatrix"),
                           (3e19, 1.0, 2.19, "dense, hot")):
        tv = tau_x2(ne, te, r)
        row = f"| {lab} | {ne:.1e} | {te} | {r} | {tv:.1f} |"
        lines.append(row)
        print(row)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("written", OUT.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
