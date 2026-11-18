# B.63 The integrated Mirnov signal through the beam interruptions

Registration: `ces_prediction/experiments/PREREGISTRATION_B63.md` with addendum 1. Runner: `ces_prediction/experiments/b63_blip_magnetics/run_b63.py`. CPU only, no model, real data (raw 2 MHz Mirnov MC1T03 and MC1T16, raw BES and ECEI of the twelve discharges with raw streams). Intervals: 95 %, 2,000 resamples of discharges; event medians. Quantities after the drift correction of addendum 1, in units of the control scatter of the discharge (median |corrected C| of its correction half in the window 0 to 3 ms after the return).

## Checks with the drift correction (addendum 1; written before any statement)

- **(a)** 41 interruptions of one absent frame, median off-time 10.50 ms: **passed**.
- **(b), repeated on the held-out half** (240 held-out pseudo-events; correction half 240): event median of corrected C per window (limit |median| <= 0.5): inside_last_3ms -0.064 [-0.536, +0.222] **passes**; return_0_3ms +0.051 [-0.360, +0.460] **passes**; return_10_13ms -0.811 [-1.137, -0.132] **fails**; return_20_23ms -0.188 [-0.795, +0.420] **passes**; return_40_43ms -0.481 [-1.429, +0.507] **passes**.
- **(c)** not recomputed (addendum 1): ratio 2.33 at the 200 BES excursions, excursion median 1.010 [0.406, 2.211] against 0.433 [0.291, 0.583] control scatters (first-run units).
- **(d)** 36 of 41 one-frame interruptions inside the raw Mirnov coverage. Per discharge (used / all; pseudo-events correction / held-out): 31097 3/3 (20/20), 31114 3/3 (20/20), 31357 3/3 (20/20), 31359 2/2 (20/20), 31747 4/4 (20/20), 31873 1/6 (20/20), 31902 7/7 (20/20), 31921 3/3 (20/20), 31923 2/2 (20/20), 32027 3/3 (20/20), 32092 4/4 (20/20), 32097 5/5 (20/20).

Control scatter per discharge (V s): 31097 1.172e-03, 31114 4.256e-05, 31357 3.346e-05, 31359 4.174e-05, 31747 3.255e-04, 31873 3.994e-05, 31902 7.706e-05, 31921 2.836e-05, 31923 4.503e-05, 32027 9.835e-05, 32092 5.195e-04, 32097 3.376e-05.

## First checks, without the drift correction (kept for the record)

- (a) passed: 41 interruptions, median off-time 10.50 ms.
- (b) **failed** (120 pseudo-events, ten per discharge): event median of C inside_last_3ms -0.453, return_0_3ms -0.187, return_10_13ms -0.221, return_20_23ms -0.455, return_40_43ms -0.638 control scatters; the last window outside 0.5. Cause (pseudo-events only): the integrated signal drifts, with a sign that is the same within a discharge and a size growing with the time since the start of the integration.
- (c) passed: ratio 2.33. (d) 36 interruptions covered.
- Code of that run: `data/.b63/run_b63_first_checks.py`; output `data/.b63/first_checks.{md,json}`.

## Statements

Each statement needs its own interval and the interval of the difference interruptions minus held-out pseudo-events (addendum 1). Units: control scatters.

| statement | interruptions | held-out pseudo-events | difference | verdict | expected |
|---|---|---|---|---|---|
| M1 C, last 3 ms before the return, differs from 0 | +1.853 [+0.656, +2.742] | -0.064 [-0.554, +0.221] | +1.918 [+0.978, +2.879] | **holds** | met, 1 to 4 |
| M2 Phi_03, 0-3 ms after the return | +25.048 [+7.051, +54.841] | -0.498 [-3.297, +0.542] | +25.546 [+7.502, +57.342] | **does not hold** | met (same sign, both exclude 0) |
| M2 Phi_16, 0-3 ms after the return | -16.057 [-45.187, -5.672] | -0.412 [-1.081, +2.300] | -15.645 [-47.349, -4.668] | | |
| M3 C decays (40_43ms minus 0-3 ms, x sign at 0-3 ms) below 0 | -2.322 [-4.113, -0.973] | -0.532 [-1.690, +0.508] | -1.790 [-3.118, -0.719] | **holds** | met (a half or more smaller) |
| M4 D, 0-3 ms after the return, includes 0 | +20.553 [+4.801, +50.788] | +0.032 [-2.503, +1.001] | +20.521 [+4.616, +52.427] | literal: **does not hold**; both intervals include 0: **does not hold** | met |

Events: 36 interruptions of one absent frame; held-out pseudo-events: 240.

## All windows (medians, control scatters)

| quantity | window | interruptions | held-out | difference |
|---|---|---|---|---|
| C | inside_last_3ms | +1.853 [+0.656, +2.742] | -0.064 [-0.554, +0.221] | +1.918 [+0.978, +2.879] |
| C | return_0_3ms | +2.289 [+1.376, +3.048] | +0.051 [-0.300, +0.451] | +2.237 [+1.381, +3.001] |
| C | return_10_13ms | +1.557 [-0.525, +1.968] | -0.811 [-1.171, -0.133] | +2.368 [+0.108, +2.751] |
| C | return_20_23ms | +1.470 [-0.448, +2.191] | -0.188 [-0.743, +0.431] | +1.659 [-0.102, +2.383] |
| C | return_40_43ms | -0.034 [-2.065, +1.503] | -0.481 [-1.418, +0.559] | +0.447 [-1.298, +1.547] |
| D | inside_last_3ms | +21.935 [+4.297, +31.857] | +0.294 [-1.488, +1.268] | +21.640 [+3.817, +33.171] |
| D | return_0_3ms | +20.553 [+4.801, +50.788] | +0.032 [-2.503, +1.001] | +20.521 [+4.616, +52.427] |
| D | return_10_13ms | +25.691 [+5.212, +60.684] | -0.146 [-1.952, +1.327] | +25.837 [+4.958, +61.167] |
| D | return_20_23ms | +28.908 [+4.675, +55.249] | -1.273 [-8.431, +1.479] | +30.181 [+4.296, +61.360] |
| D | return_40_43ms | +17.099 [+6.241, +54.310] | +1.204 [-6.787, +7.079] | +15.895 [+3.113, +51.483] |
| P03 | inside_last_3ms | +22.407 [+6.248, +34.733] | -0.059 [-1.302, +0.833] | +22.466 [+5.830, +35.868] |
| P03 | return_0_3ms | +25.048 [+7.051, +54.841] | -0.498 [-3.297, +0.542] | +25.546 [+7.502, +57.342] |
| P03 | return_10_13ms | +26.423 [+8.668, +59.831] | -0.936 [-3.814, +1.198] | +27.359 [+8.750, +64.894] |
| P03 | return_20_23ms | +35.874 [+7.088, +54.342] | -2.496 [-8.541, +1.420] | +38.370 [+7.983, +61.131] |
| P03 | return_40_43ms | +23.165 [+6.398, +51.015] | -0.340 [-7.328, +6.562] | +23.505 [+5.987, +51.315] |
| P16 | inside_last_3ms | -19.942 [-30.759, -4.115] | -0.402 [-1.715, +1.905] | -19.540 [-31.950, -2.644] |
| P16 | return_0_3ms | -16.057 [-45.187, -5.672] | -0.412 [-1.081, +2.300] | -15.645 [-47.349, -4.668] |
| P16 | return_10_13ms | -24.959 [-60.822, -4.721] | -0.719 [-2.558, +2.812] | -24.240 [-61.826, -2.949] |
| P16 | return_20_23ms | -26.891 [-54.947, -3.732] | +0.356 [-1.656, +8.697] | -27.247 [-60.296, -3.377] |
| P16 | return_40_43ms | -13.475 [-49.540, -4.583] | -3.043 [-8.518, +9.261] | -10.432 [-51.461, -0.208] |

Largest |event-median C| in the 1 ms profile: +2.423 control scatters at +1.5 ms from the return of the beam.

### 1 ms profiles (medians, control scatters; ECEI and BES levels of B.62 beside)

| ms from return | C | D | held-out C | held-out D | ECEI (B.62) | BES (B.62) |
|---|---|---|---|---|---|---|
| -45 | +nan | +nan | +nan | +nan | 0.991 | 0.998 |
| -40 | +0.159 | +0.200 | +0.073 | -0.072 | 0.992 | 1.005 |
| -35 | -0.022 | +0.118 | +0.064 | -0.386 | 0.994 | 0.996 |
| -30 | -0.174 | +0.409 | +0.060 | -0.355 | 1.000 | 0.996 |
| -25 | -0.267 | +0.404 | -0.151 | -0.222 | 0.999 | 1.008 |
| -20 | +0.207 | -0.724 | -0.034 | +0.095 | 1.002 | 1.000 |
| -15 | +0.175 | -0.004 | -0.144 | +0.496 | 1.000 | 1.003 |
| -12 | +0.303 | -0.216 | -0.066 | +0.087 | 1.001 | 1.001 |
| -10 | +0.291 | +2.281 | -0.108 | +0.194 | 0.997 | 0.137 |
| -8 | +0.539 | +5.009 | -0.267 | +0.211 | 0.990 | 0.130 |
| -6 | +1.516 | +8.679 | -0.046 | +0.280 | 0.984 | 0.127 |
| -4 | +1.366 | +11.266 | -0.063 | +0.218 | 0.980 | 0.121 |
| -2 | +1.495 | +21.647 | -0.025 | +0.485 | 0.967 | 0.121 |
| -1 | +1.808 | +21.509 | -0.002 | +0.377 | 0.964 | 0.332 |
| 0 | +1.751 | +26.390 | +0.048 | +0.465 | 0.963 | 0.575 |
| 1 | +2.423 | +21.996 | +0.107 | +0.130 | 0.952 | 0.694 |
| 2 | +1.556 | +17.630 | -0.470 | -0.014 | 0.953 | 0.748 |
| 3 | +1.962 | +20.574 | -0.017 | +0.323 | 0.947 | 0.746 |
| 5 | +1.526 | +20.041 | -0.522 | -0.082 | 0.943 | 0.730 |
| 7 | +1.895 | +21.473 | -0.429 | +0.214 | 0.931 | 0.789 |
| 10 | +1.069 | +18.027 | -0.706 | -0.072 | 0.943 | 0.895 |
| 13 | +0.084 | +25.390 | -0.604 | -0.659 | 0.937 | 0.891 |
| 15 | +0.229 | +23.150 | -0.416 | -0.155 | 0.936 | 0.891 |
| 20 | +1.344 | +32.352 | -0.209 | -0.918 | 0.948 | 0.947 |
| 25 | +1.906 | +23.483 | +0.107 | -1.389 | 0.929 | 0.946 |
| 30 | -0.008 | +27.322 | -0.585 | -0.520 | 0.945 | 0.964 |
| 40 | -0.923 | +22.949 | -1.070 | +3.203 | 0.950 | 0.963 |
| 50 | +1.182 | +17.230 | -0.054 | +1.564 | 0.953 | 0.985 |
| 60 | +0.651 | +5.863 | -0.734 | -0.543 | - | - |
| 70 | -0.059 | +6.576 | -0.321 | +0.202 | - | - |
| 79 | +0.787 | +5.169 | -0.807 | -0.071 | - | - |

## Longer interruptions (two and three absent frames; reported)

| shot | absent frames | off (ms) | C inside / 0-3 / 10-13 / 20-23 / 40-43 | D 0-3 |
|---|---|---|---|---|
| 31902 | 3 | 26.7 | +5.03 / +4.79 / +2.03 / +3.40 / +0.12 | +30.95 |
| 31902 | 3 | 26.7 | +4.98 / +5.57 / +5.11 / +3.40 / +1.63 | +74.45 |
| 31902 | 3 | 26.4 | +0.29 / +0.33 / -3.91 / -2.55 / -9.47 | +76.19 |
| 32097 | 2 | 27.2 | +2.74 / +2.73 / +0.75 / -2.47 / -5.18 | +131.13 |

Rank correlation over one-frame interruptions with an ECEI readout (n 26): corrected C at 0-3 ms after the return vs the ECEI fall 10-13 ms after it -0.387; |C| vs the fall -0.425.

## Per discharge (medians over its one-frame interruptions, control scatters)

| shot | events | C inside | C 0-3 | C 10-13 | C 40-43 | D 0-3 | Phi_03 0-3 | Phi_16 0-3 | held-out C 0-3 |
|---|---|---|---|---|---|---|---|---|---|
| 31097 | 3 | -0.05 | -0.32 | -2.12 | -9.47 | +1.44 | +1.09 | -1.78 | +0.04 |
| 31114 | 3 | +2.95 | +4.03 | +3.15 | +1.50 | +77.79 | +83.56 | -71.49 | +0.52 |
| 31357 | 3 | +2.10 | +2.57 | +3.08 | -0.98 | +33.12 | +35.69 | -30.54 | -0.00 |
| 31359 | 2 | +3.59 | +3.35 | +2.42 | -1.05 | +43.72 | +47.08 | -40.37 | +0.20 |
| 31747 | 4 | +2.33 | +2.13 | +1.96 | +3.59 | +8.15 | +10.12 | -6.17 | +0.85 |
| 31873 | 1 | +1.05 | +0.97 | -2.41 | +2.79 | +89.34 | +90.31 | -88.37 | +0.54 |
| 31902 | 4 | +3.31 | +3.00 | -1.24 | -3.61 | +15.06 | +17.20 | -12.91 | +1.09 |
| 31921 | 3 | +1.70 | +3.47 | +1.11 | +1.50 | +51.33 | +55.30 | -48.28 | +0.28 |
| 31923 | 2 | +3.54 | +2.82 | +1.68 | +0.77 | +50.47 | +53.28 | -47.65 | -0.72 |
| 32027 | 3 | +0.66 | +1.52 | -1.57 | +0.36 | +15.85 | +17.93 | -13.76 | -0.56 |
| 32092 | 4 | +0.08 | -0.40 | -0.52 | -1.58 | +3.92 | +3.24 | -3.56 | -0.82 |
| 32097 | 4 | +1.12 | +2.04 | +0.20 | -1.85 | +22.56 | +25.08 | -20.43 | -0.12 |

## Interpretations made in running the registration

1. Raw arrays are the float32 caches `data/raw_mus/.cache/s{shot}_*.npz` of the raw CSVs (spot-checked against the CSV in the first run: largest difference 4.8e-7).
2. Phi before t_off (for the profile) is the integral from t_off backward, so Phi is zero at t_off on both sides.
3. Pseudo-events: the B.62 rule (uniform in the raw window, no raw interruption and no step above 15 ms of the 10 ms file within 200 ms), inside the raw Mirnov window, length 10.5 ms; the first 40 drawn, then put in time order and split alternately.
4. Joint intervals: one draw of the twelve discharges gives the event median (over its interruptions), the held-out median (over its held-out pseudo-events) and their difference.
5. M2 is read if the window 0-3 ms passes check (b) (the check is on C); the added condition is applied to Phi_03 and Phi_16 separately.
6. M3: (median at the later window - median at 0-3 ms) x sign of the full-sample median at 0-3 ms; the added condition uses the same quantity of the held-out pseudo-events.
7. M4 (D includes zero) is a statement of no change; the added condition of addendum 1 is written for a deviation. The verdict applies the addendum literally (the interval of event minus held-out must exclude zero), and the reading that M4 is met when both intervals include zero is reported beside it.

Run time 14 s (statement stage).

## Addendum 2: relative polarity of the coils (exploratory, decided after the statements)

Exploratory check of addendum 2, decided after the statements were read; not a statement. Runner: `ces_prediction/experiments/b63_blip_magnetics/polarity_b63.py` (the registered runner is untouched). At each of the 200 BES excursions of check (c) the change of Phi_03 and of Phi_16 between the 5 ms before and the 5 ms after the excursion, with the integration and baseline of check (c) (Phi zero at the excursion, baseline the mean over the 30 ms before; the mean of the two changes reproduces the check-(c) change of C for every excursion). Changes in units of the addendum-1 control scatter of the discharge. Share of events with opposite signs, Spearman rank correlation over events, event median of Phi_16/Phi_03; 95 % intervals, 2,000 resamples of discharges.

| events | n (discharges) | share opposite signs | rank correlation | median Phi_16/Phi_03 | median Phi_03 | median Phi_16 |
|---|---|---|---|---|---|---|
| BES excursions (candidate edge bursts), change 5 ms after minus 5 ms before | 200 (11) | +0.955 [+0.906, +0.990] | -0.989 [-0.994, -0.948] | -0.947 [-0.972, -0.920] | +14.899 [+2.241, +58.708] | -14.362 [-54.252, -1.756] |
| held-out pseudo-events, same quantity at the middle of the pseudo-interruption | 240 (12) | +0.892 [+0.796, +0.963] | -0.975 [-0.992, -0.878] | -0.950 [-0.989, -0.893] | -0.429 [-2.549, +0.252] | +0.053 [-0.244, +2.188] |
| one-frame interruptions, Phi 0-3 ms after the return (t_off baseline), drift-corrected (as the statements) | 36 (12) | +0.944 [+0.872, +1.000] | -0.991 [-0.997, -0.912] | -0.859 [-0.957, -0.806] | +25.048 [+7.051, +54.841] | -16.057 [-46.734, -5.672] |
| one-frame interruptions, same, without the drift correction | 36 (12) | +1.000 [+1.000, +1.000] | -0.979 [-0.994, -0.857] | -0.885 [-0.952, -0.846] | +20.414 [+4.945, +55.779] | -16.552 [-47.490, -4.001] |

Per discharge (n, share opposite, median Phi_16/Phi_03):

| shot | excursions | held-out pseudo-events | interruptions (corrected) |
|---|---|---|---|
| 31097 | 16, 1.00, -1.26 | 20, 0.45, +0.44 | 3, 1.00, -1.60 |
| 31114 | 14, 1.00, -0.89 | 20, 0.90, -0.95 | 3, 1.00, -0.86 |
| 31357 | - | 20, 1.00, -1.06 | 3, 1.00, -0.86 |
| 31359 | 21, 1.00, -1.00 | 20, 0.95, -1.01 | 2, 1.00, -0.86 |
| 31747 | 38, 0.95, -0.95 | 20, 0.85, -0.82 | 4, 0.75, -0.56 |
| 31873 | 5, 1.00, -0.89 | 20, 1.00, -0.97 | 1, 1.00, -0.98 |
| 31902 | 8, 1.00, -0.97 | 20, 1.00, -0.95 | 4, 0.75, -0.79 |
| 31921 | 24, 1.00, -0.93 | 20, 1.00, -0.99 | 3, 1.00, -0.89 |
| 31923 | 31, 1.00, -0.95 | 20, 0.90, -0.96 | 2, 1.00, -0.88 |
| 32027 | 8, 0.88, -0.95 | 20, 1.00, -0.97 | 3, 1.00, -0.96 |
| 32092 | 13, 0.69, -0.80 | 20, 0.75, -0.55 | 4, 1.00, -1.36 |
| 32097 | 22, 0.91, -0.99 | 20, 0.90, -0.97 | 4, 1.00, -0.82 |

Rule of addendum 2: opposite effective polarity if, at the excursions, the interval of the share of opposite signs lies above 0.5 and the interval of the rank correlation below 0.

**Branch of addendum 2: the two coils have opposite effective polarity for such changes; read as exploratory: the wall field changes alike at two toroidal positions from the moment the beam goes off and for about 50 ms after its return.**

In either case no size of a displacement is given and the fall of T_i is not attributed to it.

