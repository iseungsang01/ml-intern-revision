# Estimates from published formulas, not measurements

Script: `ces_prediction/experiments/b62_beam_return/estimates_published_formulas.py`. The plasma parameters are assumed typical KSTAR edge values, not measurements of the discharges of the data set.

## (a) Beam-ion slowing down (Heidbrink & Sadler 1994, eqs. 7, 9, 10)

Inputs: deuterium beam, Z_eff = 2 (carbon), ln Lambda = 17, beam components 95, 47.5 and 31.7 keV.

| location | T_e (eV) | n_e (m^-3) | E0 (keV) | E_crit (keV) | tau_se (ms) | tau_th (ms) | share to ions |
|---|---|---|---|---|---|---|---|
| pedestal top | 500 | 2.5e+19 | 95.0 | 9.3 | 33 | 39 | 0.17 |
| pedestal top | 500 | 2.5e+19 | 47.5 | 9.3 | 33 | 28 | 0.29 |
| pedestal top | 500 | 2.5e+19 | 31.7 | 9.3 | 33 | 22 | 0.39 |
| hot pedestal top | 800 | 3.0e+19 | 95.0 | 14.9 | 56 | 53 | 0.25 |
| hot pedestal top | 800 | 3.0e+19 | 47.5 | 14.9 | 56 | 35 | 0.41 |
| hot pedestal top | 800 | 3.0e+19 | 31.7 | 14.9 | 56 | 26 | 0.52 |
| mid-radius | 1500 | 3.5e+19 | 95.0 | 28.0 | 123 | 81 | 0.39 |
| mid-radius | 1500 | 3.5e+19 | 47.5 | 28.0 | 123 | 48 | 0.58 |
| mid-radius | 1500 | 3.5e+19 | 31.7 | 28.0 | 123 | 32 | 0.69 |
| core | 2500 | 4.0e+19 | 95.0 | 46.6 | 232 | 105 | 0.53 |
| core | 2500 | 4.0e+19 | 47.5 | 46.6 | 232 | 55 | 0.72 |
| core | 2500 | 4.0e+19 | 31.7 | 46.6 | 232 | 34 | 0.81 |

## (a) Zero-dimensional balance through an interruption

dW/dt = P_h(t) - W/tau_E; P_h = beam source (fraction f_off switched off for the interruption) passed through the slowing-down kernel of a 95 keV deuteron at the parameters of the row; self-similar profiles, so dT/T = dW/W at fixed density; ohmic and other heating neglected (upper estimates). Falls are positive numbers, in per cent of the level before.

| kernel | interruption (ms) | tau_E (ms) | f_off | mean during (%) | mean over the first frame after return (%) | largest (%) | time of largest after return (ms) |
|---|---|---|---|---|---|---|---|
| pedestal top | 10.5 | 50 | 1 | 1.9 | 8.6 | 13.0 | 26 |
| pedestal top | 10.5 | 50 | 1/3 | 0.6 | 2.9 | 4.3 | 26 |
| pedestal top | 10.5 | 85 | 1 | 1.1 | 5.3 | 9.1 | 31 |
| pedestal top | 10.5 | 85 | 1/3 | 0.4 | 1.8 | 3.0 | 31 |
| pedestal top | 27.0 | 50 | 1 | 9.3 | 28.2 | 32.6 | 17 |
| pedestal top | 27.0 | 50 | 1/3 | 3.1 | 9.4 | 10.9 | 17 |
| pedestal top | 27.0 | 85 | 1 | 5.8 | 18.3 | 22.6 | 22 |
| pedestal top | 27.0 | 85 | 1/3 | 1.9 | 6.1 | 7.5 | 22 |
| mid-radius | 10.5 | 50 | 1 | 0.6 | 3.4 | 9.3 | 72 |
| mid-radius | 10.5 | 50 | 1/3 | 0.2 | 1.1 | 3.1 | 72 |
| mid-radius | 10.5 | 85 | 1 | 0.4 | 2.1 | 7.4 | 74 |
| mid-radius | 10.5 | 85 | 1/3 | 0.1 | 0.7 | 2.5 | 74 |
| mid-radius | 27.0 | 50 | 1 | 3.7 | 13.3 | 23.6 | 57 |
| mid-radius | 27.0 | 50 | 1/3 | 1.2 | 4.4 | 7.9 | 57 |
| mid-radius | 27.0 | 85 | 1 | 2.3 | 8.5 | 18.4 | 63 |
| mid-radius | 27.0 | 85 | 1/3 | 0.8 | 2.8 | 6.1 | 63 |
| core | 10.5 | 50 | 1 | 0.4 | 2.2 | 8.4 | 97 |
| core | 10.5 | 50 | 1/3 | 0.1 | 0.7 | 2.8 | 97 |
| core | 10.5 | 85 | 1 | 0.2 | 1.4 | 6.8 | 98 |
| core | 10.5 | 85 | 1/3 | 0.1 | 0.5 | 2.3 | 98 |
| core | 27.0 | 50 | 1 | 2.5 | 9.0 | 20.8 | 83 |
| core | 27.0 | 50 | 1/3 | 0.8 | 3.0 | 6.9 | 83 |
| core | 27.0 | 85 | 1 | 1.5 | 5.8 | 16.9 | 87 |
| core | 27.0 | 85 | 1/3 | 0.5 | 1.9 | 5.6 | 87 |

## (b) Optical depth of perpendicular X2 ECE (B_T = 1.8 T at R0 = 1.8 m)

Absorption of Hutchinson eqs. 5.2.39, 5.2.47, 5.2.54 with the relativistic line shape, as in syndia eceint.py (see also Zhu et al. 2019, eq. 9.3), integrated across the resonance of a uniform slab.

| case | n_e (m^-3) | T_e (keV) | R (m) | tau |
|---|---|---|---|---|
| pedestal top, dense | 2.0e+19 | 0.5 | 2.198 | 6.7 |
| pedestal top | 1.5e+19 | 0.3 | 2.198 | 2.9 |
| low density | 1.0e+19 | 0.2 | 2.22 | 1.2 |
| separatrix | 8.0e+18 | 0.1 | 2.25 | 0.5 |
| dense, hot | 3.0e+19 | 1.0 | 2.19 | 22.2 |
