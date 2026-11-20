# B.76 — the session lead's checks

Written 2026-11-20 after the batch was read, with code written independently of the implementing agent's row builder and
readout (`ces_prediction/experiments/b76_mast_availability/lead_checks/verify_b76.py`; it uses the shared MAST loaders of
`experiments/mast/` and none of the agent's code). Skill of A over B = 1 - SSE(A) / SSE(B) on identical rows; target L5.
Part C's controls are **unregistered**, on one design, and carry no interval.

`py ces_prediction/experiments/b76_mast_availability/lead_checks/verify_b76.py .`

```
(A) own pooling of the stored estimates, target L5; units: 35, M3: ['M3_q1', 'M3_q2', 'M3_q3', 'M3_q4'], M4: ['M4']
  TX over T0, D100: M1 +0.305 | M2 +0.288 | M3 +0.277 | M4 +0.443 ;  D5: M1 +0.107
  S1 MSE(T0, D100) / MSE(T0, D5), M1: 1.567 | S7 RTa over H, M1, D100: -0.487 | S10 TX over O, M1, D5: +0.346
  placebo TXp over T0, M1, D100: +0.003 | TXo over T0: +0.058
(B) 24 shots recomputed from the shot files
  rows compared: 2423 of 24 shots; largest |mine - row table|: y 0.00e+00, L5 0.00e+00, H 0.00e+00, age 0.00e+00, G 0.00e+00, th 0.00e+00, te_t 0.00e+00, te_a 0.00e+00, ne_t 0.00e+00
(C) own fits, M4, D100, L5
  fit rows 56504, test rows 24856 (shots fit 1574, test 393)
  largest |mine - stored|: T0 0.00e+00 eV, TX 0.00e+00 eV; RMSE hold 244.0, G 167.1, T0 117.9, TX 88.0, own reading 90.0; TX over T0 mine +0.443, stored +0.443
  controls (unregistered, no interval), skill over T0: Thomson scalars permuted -0.008 | T_e scalars only +0.433 | n_e scalars only +0.170 | all but the edge distance +0.461 | edge distance only -0.013
```

- A: the values of S1, S2, S4 to S7, S9, S10 and of the control with the Thomson scalars of the anchor frame equal those of `b76_mast_availability.md`.
- B: for 24 random shots (2,423 rows) the reported value, the following level, the newest visible value under D100, its age, the history scalars, the causal GP, and the Thomson T_e at frame t and at the anchor frame and n_e at frame t, recomputed from the shot files, equal the row table exactly.
- C: the lead's own fits on the chronological design reproduce the stored estimates of T0 and TX exactly.
- C, controls: with the Thomson scalars permuted over rows the skill over T0 is -0.008. With the four T_e scalars alone (present, anchor, difference, log ratio) it is +0.433 of the +0.443; with the four n_e scalars alone +0.170; without the edge distance +0.461; with the edge distance alone -0.013. The increment is carried by the electron temperature at the chord radius, not by the position of the density edge.
