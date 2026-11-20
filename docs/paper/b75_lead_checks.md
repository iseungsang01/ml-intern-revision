# B.75 — the session lead's checks

Written 2026-11-20 after the batch was read, with code written independently of the implementing agent's
(`ces_prediction/experiments/b75_availability/lead_checks/`). Parts A and B check the registered results. Parts C and D
are **unregistered** controls on one or six fit units; they carry no interval and no statement is judged on them.
Skill of A over B = 1 - SSE(A) / SSE(B) on identical rows. Target L5 throughout.

## A. Statement values from the stored estimates, own pooling; B. own inputs and fits for V4 (fit to 31991), rule D650; C. controls

`py ces_prediction/experiments/b75_availability/lead_checks/verify_b75.py .`

```
(A) from the stored estimates, own pooling
  TX over T0, L5, D10  : V1 +0.103  V2 +0.058  V3 +0.028  V4a +0.089  V4b +0.079
  TX over T0, L5, D650 : V1 +0.336  V2 +0.282  V3 +0.190  V4a +0.237  V4b +0.216
  TX over T0, L5, S    : V1 +0.327  V2 +0.293  V3 +0.231  V4a +0.028  V4b -0.254
  TE over T0, L5, D650, V1: +0.229
  T0 over G, L5, D650: needs G; see part B for V4a
  1 - MSE(TX)/MSE(own reading), L5: D10 V1 +0.538, D10 V4a +0.682, S V1 -0.003
  V1 r2 scored rows under D650 (L5): 160343; common_L5 rows: 160343
(B) own inputs for rule D650, all D1 rows
  grid keys: ['fast', 'fidx', 'grow', 'names', 'off', 'seg', 'shot', 'ti', 'time']
[  215s] history and causal GP done
  rows with a visible value: mine 187697, stored 187697
  largest |mine - stored|: H 0.00e+00 eV, age 0.00e+00 ms, G 0.00e+00 eV, history scalars 0.00e+00; anchor rows equal: True
  my D650 inputs finite on the stored common set (L5): True (160343 rows)
  V4a fit rows 127270, test rows 10146 (stored test rows 10146, same rows: True)
  V4a D650 L5: RMSE G 203.6, T0 mine 202.3 (stored 202.3), TX mine 179.1 (stored 176.7), own reading 245.3, hold 313.8
  V4a D650 L5: TX over T0 mine +0.217, stored +0.237; T0 over G mine +0.012
  |mine - stored| estimates: T0 median 0.00 eV, 99th pct 0.00; TX median 17.22 eV, 99th pct 115.99
(C) controls (unregistered), V4a, D650, L5
  placebo (fast and anchor scalars permuted over rows): TX over T0 +0.013
  fast scalars taken at the anchor row (as old as the history), 10146 test rows: over T0 -0.128; present fast scalars on the same rows: +0.205
[  225s] done
```

- A: the increments under D10, D650 and S of every design, TE over T0 (R9), and TX against the frame's own reading (R11, R12) equal the values of `b75_availability.md`.
- B: the visible values, the age, the five history scalars, the causal GP and the anchor row of all 222,672 D1 rows under D650 equal the agent's bit for bit; the lead's fit of T0 equals the stored estimates.
- B, TX: the lead's fast and anchor scalars equal the agent's up to rounding (next block), and a fit on them gives +0.217 where the stored fit gives +0.237: on the last block (10,146 rows, 75 discharges) the value moves by 0.02 under a rounding-level change of the inputs. Both lie inside the registered interval [+0.047, +0.415].
- C, placebo: with the fourteen fast inputs permuted over rows (within the fit rows and within the test rows) and the fit repeated, TX over T0 is +0.013.
- C, fast inputs as old as the history: with the eight fast scalars taken at the row of the newest visible value instead of row t (and a1 only), the skill over T0 is -0.128; with the present fast inputs on the same rows +0.205.

## B (continued). The lead's fast inputs against the agent's

`py ces_prediction/experiments/b75_availability/lead_checks/verify_b75_feats.py .`

```
largest |agent - lead| on the common set, per fast input (inputs are in standard units; a3 is a log ratio):
   BES level        8.17e-14   (sd of the input 0.964)
   BES departure    1.16e-13   (sd of the input 0.281)
   BES sd           1.76e-11   (sd of the input 0.195)
   BES increment    8.88e-14   (sd of the input 0.312)
   ECEI level       1.01e-13   (sd of the input 0.975)
   ECEI departure   1.25e-13   (sd of the input 0.191)
   ECEI sd          4.99e-07   (sd of the input 0.119)
   ECEI increment   9.59e-14   (sd of the input 0.202)
   BES a1           1.24e-14   (sd of the input 0.959)
   BES a2           1.51e-14   (sd of the input 0.335)
   BES a3           0.00e+00   (sd of the input 0.170)
   ECEI a1          7.90e-14   (sd of the input 0.950)
   ECEI a2          4.62e-14   (sd of the input 0.358)
   ECEI a3          0.00e+00   (sd of the input 0.385)
largest over all fourteen: 4.99e-07; same undefined entries: True
own fit call with the agent's fast inputs: iterations 100, largest |estimate - stored| 0.00e+00 eV, RMSE 176.7
own fit call with the lead's fast inputs: iterations 100, largest |estimate - stored| 2.73e+02 eV, RMSE 179.1
```

With the agent's inputs the lead's own fit call reproduces the stored estimates exactly (0.0 eV).

## D. Is the increment the knowledge of when the beam was last interrupted?

The time since the end of the last absent step (schedule information, known without a diagnostic; 5000 ms when the
block has none) is given to T0 and to TX. Units: the five folds of re-partition r2 (V1) and V4 (fit to 31991). The
history inputs are the agent's cached ones (checked in B); the fast inputs are the lead's.

`py ces_prediction/experiments/b75_availability/lead_checks/verify_b75_blip.py .`

```
D1 rows: share within 100 ms after an absent step 0.061, within 300 ms 0.173, none in the block 0.311
  D650  V1 r2 (5 folds)        TX over T0 +0.336 | with the time since the last absent step in both: +0.335 | that time alone over T0: +0.023
        rows within 100 ms after an absent step: TX over T0 +0.448, with the time in both +0.429; later rows: +0.320, +0.323
  D650  V4a                    TX over T0 +0.217 | with the time since the last absent step in both: +0.204 | that time alone over T0: +0.035
        rows within 100 ms after an absent step: TX over T0 +0.414, with the time in both +0.532; later rows: +0.202, +0.169
  S     V1 r2 (5 folds)        TX over T0 +0.314 | with the time since the last absent step in both: +0.310 | that time alone over T0: +0.014
        rows within 100 ms after an absent step: TX over T0 +0.200, with the time in both +0.199; later rows: +0.323, +0.319
  S     V4a                    TX over T0 +0.045 | with the time since the last absent step in both: -0.009 | that time alone over T0: +0.002
        rows within 100 ms after an absent step: TX over T0 +0.134, with the time in both +0.108; later rows: +0.041, -0.015
```

- That time alone adds +0.023 to T0 under V1, D650 (+0.035 under V4; +0.014 under V1, S).
- With it in both estimators the increment is +0.335 under V1, D650 (without: +0.336), +0.204 under V4, D650 (+0.217), +0.310 under V1, S (+0.314), -0.009 under V4, S (+0.045).
- By that time (a stratum on an input): rows within 100 ms after an absent step +0.448, later rows +0.320 (V1, D650); under V1, S +0.200 and +0.323.

Reading: the increment under a stale history is not the timing of the beam interruptions, it needs the fast samples of the present, and it disappears when the fast inputs are permuted.
