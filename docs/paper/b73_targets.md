# B.73 — what the fast channels add: by target, by validation design, without a network

Registered: `ces_prediction/experiments/PREREGISTRATION_B73.md`, with Addendum 1 of the session lead (strata on the inputs, Q11-Q13) and Addendum 2 of the implementing agent (implementation choices), both written before any readout. CPU only; no network trained or run. T_i, cut population, the D1 rows of B.70 (222,672 rows, 635 discharges). Exploratory on used data (registration section 1): the reviewers had read forms of several of these numbers on the same discharges.

Conventions. Skill of A over B = 1 - SSE(A) / SSE(B) on identical rows; MSE reduction in % = 100 x skill. Increment = the estimator with fast inputs over the same estimator without them. Each comparison: point, then 95 % intervals with discharges resampled (d) and with shot clusters resampled (c), 20,000 draws (V1: a re-partition drawn first; V2: an assignment drawn first; V3, V4: the scored units). Verdicts on the cluster form. Point under V1 and V2 = mean over the three re-partitions or assignments. Checks: `docs/paper/b73_checks.md` (all pass).

Order of work. The strata of Addendum 1 (`strata_inputs.py`) were computed before the first run of this readout. Before them, these results existed: checks (a), (b), (c) and (e), and the filter parameters printed in the fit log. No increment, skill, AUC or statement had been computed.

## Statements (cluster interval)

| Id | statement | value: point, d interval, c interval | verdict | registered expectation | note |
|---|---|---|---|---|---|
| Q1 | V1, as built: increment of the two-stage estimator over its twin against L5 is less than half of that against F (interval of the ratio of the two MSE reductions below 0.5) | +0.116; d [-0.155, +0.310]; c [-0.267, +0.403] | **holds** | holds (0.1 to 0.4) |  |
| Q2 | V1, target L5, fitted to L5: TX over T0 above zero | +0.075; d [+0.043, +0.095]; c [+0.039, +0.106] | **holds** | holds (+0.03 to +0.07) |  |
| Q3 | V2, target L5: TX over T0 above zero | +0.057; d [+0.043, +0.070]; c [+0.034, +0.079] | **holds** | holds (+0.02 to +0.06) |  |
| Q4 | V3, target L5: TX over T0 above zero | +0.031; d [+0.012, +0.049]; c [-0.006, +0.060] | **does not hold** | uncertain (+0.01 to +0.04) |  |
| Q5 | V4 (fit to 31991, score from 32312), target L5: TX over T0 above zero | +0.084; d [+0.053, +0.115]; c [+0.048, +0.123] | **holds** | uncertain (0.00 to +0.03) | V4b (fit to 32310): +0.071; c [+0.051, +0.087] |
| Q6 | V1, target L5: TE over T0 above zero (needs neither the beam nor the spectrometer) | +0.032; d [+0.021, +0.043]; c [+0.022, +0.043] | **holds** | uncertain (0.00 to +0.03) |  |
| Q7 | V1, target L5: TX[L5] not worse than the gain-fitted two-stage estimator (interval of its skill over it not entirely below zero) | +0.065; d [+0.034, +0.087]; c [+0.031, +0.095] | **holds** | holds | interval above zero: True |
| Q8 | V4 (fit to 31991), target L5: T0[L5] over G above zero | +0.254; d [+0.208, +0.301]; c [+0.214, +0.307] | **holds** | holds | V4b: +0.312; c [+0.251, +0.378] |
| Q9-V1 | V1: TX over T0 against F (fitted to F) exceeds TX over T0 against L5 (fitted to L5), on the L5 rows | +0.025; d [+0.003, +0.056]; c [-0.012, +0.064] | **does not hold** | holds |  |
| Q9-V4 | V4 (fit to 31991): the same | -0.065; d [-0.100, -0.031]; c [-0.115, -0.017] | **does not hold** | uncertain | V4b: -0.033; c [-0.055, -0.011] |
| Q10 | V4 (fit to 31991): AUC with the fast scalars minus without above zero | +0.023; d [+0.013, +0.036]; c [+0.012, +0.036] | **holds** | holds (+0.01 to +0.05) | V4b: +0.027; c [+0.013, +0.046] |
| Q11 | (Addendum 1) V1, class r_B above 1.03: TX over T0 above zero against F and not above zero against L5 | against_F: +0.099; d [+0.082, +0.116]; c [+0.068, +0.133]<br>against_L5: +0.065; d [+0.034, +0.089]; c [+0.027, +0.098] | **does not hold** | holds |  |
| Q12 | (Addendum 1) V1, class r_B below 0.9: TX over T0 against L5 above zero | +0.183; d [+0.139, +0.227]; c [+0.124, +0.254] | **holds** | holds (+0.05 to +0.15) |  |
| Q13 | (Addendum 1) log-log slope of T_i on r_B above 0.5, and the interval of the slope of fresh V_rot within +-0.1 (discharges resampled) | T_i: +1.285; d [+1.147, +1.422]<br>V_rot: +0.024; d [-0.009, +0.058] | **holds** | holds | rows 64,822, discharges 371 |

Q1 is a ratio of MSE reductions (L5 over F); Q9 a difference of skills (F minus L5); Q10 a difference of areas under the ROC curve. Q7 reads 'not below zero' as 'the interval does not lie entirely below zero' (Addendum 2 section 8). Q13 is judged on discharge intervals (Addendum 1).

## The fast-channel increment, design by target

Rows = design; columns = learner; one panel per target. Each cell: MSE skill (MSE reduction in %), d interval, c interval. Network-based rows exist under V1 only; TB, TE, TXlag under V1 and V4 only; filters for F and L5 only. Each fitted estimator is fitted to the target it is scored on (as-built network rows are not fitted to any target).

### Target F

| design | rows / discharges / clusters | trees TX over T0 | filters KX over K0 | TB over T0 (BES only) | TE over T0 (ECEI only) | TXlag over T0 (one row old) | network over twin, as built | two-stage over two-stage twin, as built | network over twin, gain fitted | two-stage over two-stage twin, gain fitted |
|---|---|---|---|---|---|---|---|---|---|---|
| V1 random (r2-r4) | 222,672 / 635 / 86 | +0.087 (+8.7 %)<br>d [+0.073, +0.102]<br>c [+0.059, +0.122] | +0.050 (+5.0 %)<br>d [+0.041, +0.059]<br>c [+0.032, +0.070] | +0.063 (+6.3 %)<br>d [+0.054, +0.073]<br>c [+0.047, +0.084] | +0.053 (+5.3 %)<br>d [+0.039, +0.067]<br>c [+0.027, +0.086] | +0.016 (+1.6 %)<br>d [+0.010, +0.021]<br>c [+0.008, +0.024] | +0.052 (+5.2 %)<br>d [+0.029, +0.075]<br>c [+0.019, +0.084] | +0.097 (+9.7 %)<br>d [+0.078, +0.116]<br>c [+0.064, +0.137] | +0.064 (+6.4 %)<br>d [+0.050, +0.080]<br>c [+0.042, +0.088] | +0.099 (+9.9 %)<br>d [+0.083, +0.116]<br>c [+0.069, +0.135] |
| V2 cluster-blocked (3 assignments) | 222,672 / 635 / 86 | +0.072 (+7.2 %)<br>d [+0.057, +0.087]<br>c [+0.048, +0.102] | +0.048 (+4.8 %)<br>d [+0.039, +0.057]<br>c [+0.031, +0.068] | - | - | - | - | - | - | - |
| V3 rolling origin (fifths 2-5 pooled) | 161,744 / 508 / 65 | +0.045 (+4.5 %)<br>d [+0.032, +0.058]<br>c [+0.023, +0.072] | +0.041 (+4.1 %)<br>d [+0.032, +0.050]<br>c [+0.023, +0.060] | - | - | - | - | - | - | - |
| V4 chronological: fit to 31991, score from 32312 | 20,062 / 95 / 19 | +0.019 (+1.9 %)<br>d [+0.002, +0.038]<br>c [+0.001, +0.036] | +0.027 (+2.7 %)<br>d [+0.016, +0.038]<br>c [+0.006, +0.050] | +0.021 (+2.1 %)<br>d [+0.004, +0.037]<br>c [+0.004, +0.031] | +0.019 (+1.9 %)<br>d [+0.002, +0.036]<br>c [-0.000, +0.035] | +0.000 (+0.0 %)<br>d [-0.013, +0.012]<br>c [-0.014, +0.014] | - | - | - | - |
| V4 chronological: fit to 32310, score from 32312 | 20,062 / 95 / 19 | +0.031 (+3.1 %)<br>d [+0.013, +0.049]<br>c [+0.010, +0.045] | +0.027 (+2.7 %)<br>d [+0.016, +0.038]<br>c [+0.006, +0.051] | +0.036 (+3.6 %)<br>d [+0.019, +0.052]<br>c [+0.016, +0.053] | +0.004 (+0.4 %)<br>d [-0.009, +0.017]<br>c [-0.012, +0.019] | +0.010 (+1.0 %)<br>d [-0.001, +0.021]<br>c [-0.005, +0.024] | - | - | - | - |

### Target N1

| design | rows / discharges / clusters | trees TX over T0 | network over twin, as built | two-stage over two-stage twin, as built | network over twin, gain fitted | two-stage over two-stage twin, gain fitted |
|---|---|---|---|---|---|---|
| V1 random (r2-r4) | 214,768 / 627 / 85 | +0.042 (+4.2 %)<br>d [+0.034, +0.051]<br>c [+0.027, +0.060] | +0.021 (+2.1 %)<br>d [+0.002, +0.039]<br>c [-0.007, +0.046] | +0.028 (+2.8 %)<br>d [+0.014, +0.042]<br>c [+0.006, +0.053] | +0.045 (+4.5 %)<br>d [+0.035, +0.055]<br>c [+0.029, +0.061] | +0.049 (+4.9 %)<br>d [+0.040, +0.058]<br>c [+0.034, +0.067] |

### Target L3

| design | rows / discharges / clusters | trees TX over T0 | network over twin, as built | two-stage over two-stage twin, as built | network over twin, gain fitted | two-stage over two-stage twin, gain fitted |
|---|---|---|---|---|---|---|
| V1 random (r2-r4) | 204,733 / 611 / 85 | +0.081 (+8.1 %)<br>d [+0.066, +0.096]<br>c [+0.055, +0.110] | +0.035 (+3.5 %)<br>d [+0.006, +0.062]<br>c [-0.009, +0.073] | +0.038 (+3.8 %)<br>d [+0.017, +0.059]<br>c [+0.005, +0.073] | +0.075 (+7.5 %)<br>d [+0.060, +0.090]<br>c [+0.052, +0.096] | +0.077 (+7.7 %)<br>d [+0.065, +0.089]<br>c [+0.058, +0.098] |

### Target L5

| design | rows / discharges / clusters | trees TX over T0 | filters KX over K0 | TB over T0 (BES only) | TE over T0 (ECEI only) | TXlag over T0 (one row old) | network over twin, as built | two-stage over two-stage twin, as built | network over twin, gain fitted | two-stage over two-stage twin, gain fitted |
|---|---|---|---|---|---|---|---|---|---|---|
| V1 random (r2-r4) | 215,369 / 613 / 85 | +0.075 (+7.5 %)<br>d [+0.043, +0.095]<br>c [+0.039, +0.106] | +0.055 (+5.5 %)<br>d [+0.044, +0.066]<br>c [+0.035, +0.080] | +0.065 (+6.5 %)<br>d [+0.040, +0.082]<br>c [+0.035, +0.093] | +0.032 (+3.2 %)<br>d [+0.021, +0.043]<br>c [+0.022, +0.043] | +0.033 (+3.3 %)<br>d [+0.009, +0.048]<br>c [+0.007, +0.054] | +0.016 (+1.6 %)<br>d [-0.020, +0.045]<br>c [-0.036, +0.054] | +0.013 (+1.3 %)<br>d [-0.016, +0.035]<br>c [-0.025, +0.045] | +0.062 (+6.2 %)<br>d [+0.048, +0.077]<br>c [+0.040, +0.080] | +0.062 (+6.2 %)<br>d [+0.050, +0.073]<br>c [+0.046, +0.077] |
| V2 cluster-blocked (3 assignments) | 215,369 / 613 / 85 | +0.057 (+5.7 %)<br>d [+0.043, +0.070]<br>c [+0.034, +0.079] | +0.054 (+5.4 %)<br>d [+0.044, +0.066]<br>c [+0.035, +0.078] | - | - | - | - | - | - | - |
| V3 rolling origin (fifths 2-5 pooled) | 155,475 / 486 / 64 | +0.031 (+3.1 %)<br>d [+0.012, +0.049]<br>c [-0.006, +0.060] | +0.042 (+4.2 %)<br>d [+0.029, +0.055]<br>c [+0.021, +0.067] | - | - | - | - | - | - | - |
| V4 chronological: fit to 31991, score from 32312 | 19,156 / 83 / 19 | +0.084 (+8.4 %)<br>d [+0.053, +0.115]<br>c [+0.048, +0.123] | -0.007 (-0.7 %)<br>d [-0.026, +0.009]<br>c [-0.038, +0.010] | +0.063 (+6.3 %)<br>d [+0.038, +0.087]<br>c [+0.037, +0.090] | +0.058 (+5.8 %)<br>d [+0.033, +0.084]<br>c [+0.030, +0.082] | +0.041 (+4.1 %)<br>d [+0.012, +0.070]<br>c [+0.008, +0.080] | - | - | - | - |
| V4 chronological: fit to 32310, score from 32312 | 19,156 / 83 / 19 | +0.071 (+7.1 %)<br>d [+0.052, +0.089]<br>c [+0.051, +0.087] | -0.005 (-0.5 %)<br>d [-0.022, +0.010]<br>c [-0.033, +0.011] | +0.052 (+5.2 %)<br>d [+0.032, +0.072]<br>c [+0.030, +0.071] | +0.048 (+4.8 %)<br>d [+0.029, +0.064]<br>c [+0.031, +0.063] | +0.042 (+4.2 %)<br>d [+0.027, +0.057]<br>c [+0.020, +0.060] | - | - | - | - |

### Target L10

| design | rows / discharges / clusters | trees TX over T0 | network over twin, as built | two-stage over two-stage twin, as built | network over twin, gain fitted | two-stage over two-stage twin, gain fitted |
|---|---|---|---|---|---|---|
| V1 random (r2-r4) | 215,667 / 611 / 85 | +0.074 (+7.4 %)<br>d [+0.037, +0.096]<br>c [+0.037, +0.101] | +0.004 (+0.4 %)<br>d [-0.030, +0.034]<br>c [-0.045, +0.046] | -0.015 (-1.5 %)<br>d [-0.042, +0.007]<br>c [-0.047, +0.016] | +0.056 (+5.6 %)<br>d [+0.042, +0.071]<br>c [+0.037, +0.074] | +0.047 (+4.7 %)<br>d [+0.038, +0.057]<br>c [+0.036, +0.060] |

## Against F and against L5 on identical rows (the L5 rows)

Each learner fitted to F scored against F, and fitted to L5 scored against L5, on the rows where L5 is defined. Ratio = MSE reduction against L5 / that against F; difference = skill against F minus skill against L5.

| design | learner | against F | against L5 | ratio L5 / F | difference F - L5 |
|---|---|---|---|---|---|
| V1 random (r2-r4) | trees TX/T0 | +0.100 (+10.0 %)<br>d [+0.084, +0.116]<br>c [+0.069, +0.139] | +0.075 (+7.5 %)<br>d [+0.043, +0.095]<br>c [+0.039, +0.106] | +0.755<br>d [+0.439, +0.970]<br>c [+0.429, +1.141] | +0.025<br>d [+0.003, +0.056]<br>c [-0.012, +0.064] |
| V1 random (r2-r4) | filters KX/K0 | +0.063 (+6.3 %)<br>d [+0.053, +0.073]<br>c [+0.044, +0.084] | +0.055 (+5.5 %)<br>d [+0.044, +0.066]<br>c [+0.035, +0.080] | +0.872<br>d [+0.736, +1.019]<br>c [+0.645, +1.128] | +0.008<br>d [-0.001, +0.017]<br>c [-0.008, +0.023] |
| V1 random (r2-r4) | network c/t as built | +0.063 (+6.3 %)<br>d [+0.038, +0.087]<br>c [+0.025, +0.097] | +0.016 (+1.6 %)<br>d [-0.020, +0.045]<br>c [-0.036, +0.054] | +0.248<br>d [-0.475, +0.595]<br>c [-1.181, +0.727] | +0.047<br>d [+0.027, +0.069]<br>c [+0.016, +0.079] |
| V1 random (r2-r4) | two-stage c2/t2 as built | +0.111 (+11.1 %)<br>d [+0.090, +0.132]<br>c [+0.073, +0.154] | +0.013 (+1.3 %)<br>d [-0.016, +0.035]<br>c [-0.025, +0.045] | +0.116<br>d [-0.155, +0.310]<br>c [-0.267, +0.403] | +0.098<br>d [+0.072, +0.126]<br>c [+0.056, +0.143] |
| V1 random (r2-r4) | network c/t gain-fitted | +0.074 (+7.4 %)<br>d [+0.059, +0.092]<br>c [+0.050, +0.100] | +0.062 (+6.2 %)<br>d [+0.048, +0.077]<br>c [+0.040, +0.080] | +0.840<br>d [+0.712, +0.983]<br>c [+0.657, +1.096] | +0.012<br>d [+0.001, +0.023]<br>c [-0.006, +0.029] |
| V1 random (r2-r4) | two-stage c2/t2 gain-fitted | +0.112 (+11.2 %)<br>d [+0.094, +0.131]<br>c [+0.078, +0.151] | +0.062 (+6.2 %)<br>d [+0.050, +0.073]<br>c [+0.046, +0.077] | +0.551<br>d [+0.450, +0.671]<br>c [+0.413, +0.753] | +0.050<br>d [+0.033, +0.068]<br>c [+0.021, +0.085] |
| V1 random (r2-r4) | TE/T0 | +0.062 (+6.2 %)<br>d [+0.047, +0.078]<br>c [+0.032, +0.100] | +0.032 (+3.2 %)<br>d [+0.021, +0.043]<br>c [+0.022, +0.043] | +0.522<br>d [+0.348, +0.738]<br>c [+0.313, +1.006] | +0.030<br>d [+0.014, +0.046]<br>c [-0.000, +0.065] |
| V1 random (r2-r4) | TB/T0 | +0.071 (+7.1 %)<br>d [+0.060, +0.081]<br>c [+0.052, +0.093] | +0.065 (+6.5 %)<br>d [+0.040, +0.082]<br>c [+0.035, +0.093] | +0.923<br>d [+0.597, +1.159]<br>c [+0.547, +1.318] | +0.005<br>d [-0.011, +0.028]<br>c [-0.020, +0.032] |
| V1 random (r2-r4) | TXlag/T0 | +0.018 (+1.8 %)<br>d [+0.012, +0.024]<br>c [+0.010, +0.026] | +0.033 (+3.3 %)<br>d [+0.009, +0.048]<br>c [+0.007, +0.054] | +1.851<br>d [+0.588, +2.958]<br>c [+0.455, +3.150] | -0.015<br>d [-0.030, +0.007]<br>c [-0.033, +0.007] |
| V2 cluster-blocked (3 assignments) | trees TX/T0 | +0.083 (+8.3 %)<br>d [+0.066, +0.099]<br>c [+0.056, +0.116] | +0.057 (+5.7 %)<br>d [+0.043, +0.070]<br>c [+0.034, +0.079] | +0.694<br>d [+0.522, +0.896]<br>c [+0.460, +1.024] | +0.025<br>d [+0.007, +0.043]<br>c [-0.002, +0.053] |
| V2 cluster-blocked (3 assignments) | filters KX/K0 | +0.061 (+6.1 %)<br>d [+0.052, +0.071]<br>c [+0.043, +0.083] | +0.054 (+5.4 %)<br>d [+0.044, +0.066]<br>c [+0.035, +0.078] | +0.886<br>d [+0.749, +1.037]<br>c [+0.672, +1.129] | +0.007<br>d [-0.002, +0.016]<br>c [-0.007, +0.021] |
| V3 rolling origin (fifths 2-5 pooled) | trees TX/T0 | +0.052 (+5.2 %)<br>d [+0.039, +0.066]<br>c [+0.026, +0.082] | +0.031 (+3.1 %)<br>d [+0.012, +0.049]<br>c [-0.006, +0.060] | +0.590<br>d [+0.237, +0.975]<br>c [-0.135, +1.420] | +0.021<br>d [+0.001, +0.042]<br>c [-0.014, +0.057] |
| V3 rolling origin (fifths 2-5 pooled) | filters KX/K0 | +0.054 (+5.4 %)<br>d [+0.044, +0.063]<br>c [+0.036, +0.073] | +0.042 (+4.2 %)<br>d [+0.029, +0.055]<br>c [+0.021, +0.067] | +0.787<br>d [+0.584, +0.993]<br>c [+0.497, +1.086] | +0.011<br>d [+0.000, +0.022]<br>c [-0.005, +0.026] |
| V4 chronological: fit to 31991, score from 32312 | trees TX/T0 | +0.019 (+1.9 %)<br>d [+0.004, +0.036]<br>c [+0.002, +0.035] | +0.084 (+8.4 %)<br>d [+0.053, +0.115]<br>c [+0.048, +0.123] | +4.438<br>d [+1.980, +17.998]<br>c [+1.312, +25.440] | -0.065<br>d [-0.100, -0.031]<br>c [-0.115, -0.017] |
| V4 chronological: fit to 31991, score from 32312 | filters KX/K0 | +0.033 (+3.3 %)<br>d [+0.021, +0.045]<br>c [+0.011, +0.057] | -0.007 (-0.7 %)<br>d [-0.026, +0.009]<br>c [-0.038, +0.010] | -0.221<br>d [-0.939, +0.240]<br>c [-2.760, +0.204] | +0.040<br>d [+0.024, +0.060]<br>c [+0.025, +0.061] |
| V4 chronological: fit to 31991, score from 32312 | TE/T0 | +0.018 (+1.8 %)<br>d [+0.003, +0.035]<br>c [-0.004, +0.034] | +0.058 (+5.8 %)<br>d [+0.033, +0.084]<br>c [+0.030, +0.082] | +3.188<br>d [+1.692, +13.393]<br>c [-11.347, +18.236] | -0.040<br>d [-0.062, -0.019]<br>c [-0.061, -0.016] |
| V4 chronological: fit to 31991, score from 32312 | TB/T0 | +0.021 (+2.1 %)<br>d [+0.004, +0.038]<br>c [+0.004, +0.031] | +0.063 (+6.3 %)<br>d [+0.038, +0.087]<br>c [+0.037, +0.090] | +2.927<br>d [+1.346, +11.317]<br>c [+1.600, +9.526] | -0.041<br>d [-0.068, -0.013]<br>c [-0.069, -0.018] |
| V4 chronological: fit to 31991, score from 32312 | TXlag/T0 | +0.000 (+0.0 %)<br>d [-0.011, +0.011]<br>c [-0.015, +0.012] | +0.041 (+4.1 %)<br>d [+0.012, +0.070]<br>c [+0.008, +0.080] | +122.760<br>d [-120.952, +110.042]<br>c [-104.235, +96.405] | -0.040<br>d [-0.072, -0.010]<br>c [-0.088, -0.001] |
| V4 chronological: fit to 32310, score from 32312 | trees TX/T0 | +0.038 (+3.8 %)<br>d [+0.020, +0.055]<br>c [+0.016, +0.053] | +0.071 (+7.1 %)<br>d [+0.052, +0.089]<br>c [+0.051, +0.087] | +1.880<br>d [+1.257, +3.541]<br>c [+1.235, +3.915] | -0.033<br>d [-0.055, -0.012]<br>c [-0.055, -0.011] |
| V4 chronological: fit to 32310, score from 32312 | filters KX/K0 | +0.033 (+3.3 %)<br>d [+0.021, +0.045]<br>c [+0.012, +0.058] | -0.005 (-0.5 %)<br>d [-0.022, +0.010]<br>c [-0.033, +0.011] | -0.158<br>d [-0.822, +0.270]<br>c [-2.441, +0.230] | +0.038<br>d [+0.022, +0.056]<br>c [+0.023, +0.058] |
| V4 chronological: fit to 32310, score from 32312 | TE/T0 | +0.003 (+0.3 %)<br>d [-0.010, +0.017]<br>c [-0.016, +0.018] | +0.048 (+4.8 %)<br>d [+0.029, +0.064]<br>c [+0.031, +0.063] | +14.066<br>d [-99.818, +100.725]<br>c [-79.397, +87.217] | -0.045<br>d [-0.065, -0.022]<br>c [-0.072, -0.024] |
| V4 chronological: fit to 32310, score from 32312 | TB/T0 | +0.043 (+4.3 %)<br>d [+0.025, +0.060]<br>c [+0.021, +0.062] | +0.052 (+5.2 %)<br>d [+0.032, +0.072]<br>c [+0.030, +0.071] | +1.217<br>d [+0.720, +2.144]<br>c [+0.807, +2.076] | -0.009<br>d [-0.033, +0.014]<br>c [-0.028, +0.009] |
| V4 chronological: fit to 32310, score from 32312 | TXlag/T0 | +0.012 (+1.2 %)<br>d [+0.002, +0.023]<br>c [-0.003, +0.024] | +0.042 (+4.2 %)<br>d [+0.027, +0.057]<br>c [+0.020, +0.060] | +3.598<br>d [+1.486, +17.239]<br>c [-20.590, +30.021] | -0.030<br>d [-0.048, -0.011]<br>c [-0.052, -0.004] |

## Every estimator: skill over G and over H, RMSE, median absolute error

Points of RMSE and median absolute error: mean over re-partitions (V1) or assignments (V2). Estimator names: T0, TX, TB, TE, TXlag [target fitted to]; K0, KX filters [target fitted to]; G causal GP; H hold; M5 median of the last five values; c, t, c2, t2 the stored network-based estimates (V1).

### V1 random (r2-r4), target F

| estimator | skill over G | skill over H | RMSE (eV) | median abs. error (eV) |
|---|---|---|---|---|
| T0[F] | +0.099<br>d [+0.090, +0.108]<br>c [+0.087, +0.112] | +0.382<br>d [+0.368, +0.394]<br>c [+0.360, +0.398] | 182.9 | 41.7 |
| TX[F] | +0.177<br>d [+0.161, +0.194]<br>c [+0.148, +0.212] | +0.435<br>d [+0.420, +0.450]<br>c [+0.412, +0.459] | 174.8 | 39.7 |
| TB[F] | +0.156<br>d [+0.143, +0.170]<br>c [+0.135, +0.180] | +0.421<br>d [+0.407, +0.434]<br>c [+0.402, +0.437] | 177.0 | 40.3 |
| TE[F] | +0.146<br>d [+0.131, +0.163]<br>c [+0.119, +0.179] | +0.414<br>d [+0.398, +0.429]<br>c [+0.386, +0.440] | 178.0 | 40.5 |
| TXlag[F] | +0.113<br>d [+0.102, +0.124]<br>c [+0.098, +0.130] | +0.391<br>d [+0.377, +0.404]<br>c [+0.371, +0.407] | 181.5 | 41.3 |
| K0[F] | +0.004<br>d [-0.009, +0.017]<br>c [-0.014, +0.021] | +0.316<br>d [+0.300, +0.331]<br>c [+0.284, +0.340] | 192.3 | 43.3 |
| KX[F] | +0.053<br>d [+0.039, +0.067]<br>c [+0.033, +0.072] | +0.350<br>d [+0.336, +0.364]<br>c [+0.324, +0.369] | 187.5 | 41.8 |
| network, gain fitted [F] | +0.104<br>d [+0.087, +0.121]<br>c [+0.084, +0.126] | +0.385<br>d [+0.369, +0.401]<br>c [+0.359, +0.405] | 182.4 | 43.9 |
| two-stage, gain fitted [F] | +0.178<br>d [+0.160, +0.196]<br>c [+0.150, +0.210] | +0.436<br>d [+0.420, +0.451]<br>c [+0.412, +0.459] | 174.7 | 40.8 |
| G causal GP | - | +0.314<br>d [+0.297, +0.329]<br>c [+0.285, +0.336] | 192.7 | 44.1 |
| H hold | -0.457<br>d [-0.490, -0.423]<br>c [-0.505, -0.398] | - | 232.6 | 45.4 |
| M5 median of last five | -0.104<br>d [-0.125, -0.084]<br>c [-0.142, -0.073] | +0.242<br>d [+0.218, +0.265]<br>c [+0.190, +0.280] | 202.5 | 44.4 |
| network (first stage), as built | +0.093<br>d [+0.071, +0.113]<br>c [+0.063, +0.121] | +0.378<br>d [+0.358, +0.395]<br>c [+0.346, +0.401] | 183.5 | 45.7 |
| history-only twin, as built | +0.043<br>d [+0.024, +0.060]<br>c [+0.020, +0.062] | +0.343<br>d [+0.324, +0.360]<br>c [+0.312, +0.366] | 188.5 | 45.4 |
| two-stage estimator, as built | +0.176<br>d [+0.156, +0.196]<br>c [+0.146, +0.211] | +0.435<br>d [+0.418, +0.450]<br>c [+0.409, +0.459] | 174.9 | 41.2 |
| two-stage twin, as built | +0.088<br>d [+0.076, +0.099]<br>c [+0.075, +0.101] | +0.374<br>d [+0.358, +0.388]<br>c [+0.350, +0.392] | 184.1 | 43.0 |
| skill: TX over c2 gain-fitted | -0.001<br>d [-0.012, +0.011]<br>c [-0.014, +0.015] | | | |
| skill: TX over c2 as built | +0.001<br>d [-0.011, +0.016]<br>c [-0.013, +0.020] | | | |
| skill: T0 over t2 as built | +0.012<br>d [+0.003, +0.024]<br>c [+0.000, +0.028] | | | |

### V1 random (r2-r4), target N1

| estimator | skill over G | skill over H | RMSE (eV) | median abs. error (eV) |
|---|---|---|---|---|
| T0[N1] | +0.079<br>d [+0.068, +0.089]<br>c [+0.063, +0.095] | +0.340<br>d [+0.326, +0.354]<br>c [+0.316, +0.361] | 176.3 | 44.9 |
| TX[N1] | +0.118<br>d [+0.105, +0.130]<br>c [+0.098, +0.141] | +0.368<br>d [+0.355, +0.381]<br>c [+0.349, +0.386] | 172.5 | 43.7 |
| network, gain fitted [N1] | +0.073<br>d [+0.056, +0.089]<br>c [+0.050, +0.097] | +0.336<br>d [+0.320, +0.352]<br>c [+0.312, +0.358] | 176.9 | 46.2 |
| two-stage, gain fitted [N1] | +0.115<br>d [+0.102, +0.129]<br>c [+0.094, +0.140] | +0.367<br>d [+0.352, +0.381]<br>c [+0.346, +0.387] | 172.7 | 44.0 |
| G causal GP | - | +0.284<br>d [+0.269, +0.298]<br>c [+0.257, +0.306] | 183.7 | 46.4 |
| H hold | -0.397<br>d [-0.425, -0.368]<br>c [-0.442, -0.345] | - | 217.1 | 47.8 |
| M5 median of last five | -0.119<br>d [-0.143, -0.097]<br>c [-0.159, -0.083] | +0.199<br>d [+0.172, +0.223]<br>c [+0.145, +0.242] | 194.3 | 46.2 |
| network (first stage), as built | +0.050<br>d [+0.029, +0.069]<br>c [+0.022, +0.078] | +0.320<br>d [+0.300, +0.337]<br>c [+0.292, +0.344] | 179.1 | 48.9 |
| history-only twin, as built | +0.029<br>d [+0.011, +0.047]<br>c [+0.004, +0.056] | +0.305<br>d [+0.286, +0.323]<br>c [+0.273, +0.334] | 181.0 | 47.2 |
| two-stage estimator, as built | +0.097<br>d [+0.080, +0.113]<br>c [+0.071, +0.126] | +0.353<br>d [+0.337, +0.369]<br>c [+0.330, +0.375] | 174.6 | 45.3 |
| two-stage twin, as built | +0.070<br>d [+0.056, +0.084]<br>c [+0.050, +0.092] | +0.334<br>d [+0.318, +0.350]<br>c [+0.307, +0.359] | 177.1 | 45.2 |
| skill: TX over c2 gain-fitted | +0.002<br>d [-0.007, +0.012]<br>c [-0.010, +0.014] | | | |
| skill: TX over c2 as built | +0.023<br>d [+0.011, +0.036]<br>c [+0.007, +0.039] | | | |
| skill: T0 over t2 as built | +0.009<br>d [-0.001, +0.019]<br>c [-0.005, +0.022] | | | |

### V1 random (r2-r4), target L3

| estimator | skill over G | skill over H | RMSE (eV) | median abs. error (eV) |
|---|---|---|---|---|
| T0[L3] | +0.126<br>d [+0.112, +0.141]<br>c [+0.103, +0.150] | +0.480<br>d [+0.461, +0.498]<br>c [+0.443, +0.512] | 128.5 | 37.4 |
| TX[L3] | +0.197<br>d [+0.179, +0.214]<br>c [+0.172, +0.225] | +0.522<br>d [+0.506, +0.537]<br>c [+0.494, +0.545] | 123.2 | 36.3 |
| network, gain fitted [L3] | +0.101<br>d [+0.075, +0.126]<br>c [+0.059, +0.141] | +0.465<br>d [+0.443, +0.486]<br>c [+0.427, +0.500] | 130.3 | 38.3 |
| two-stage, gain fitted [L3] | +0.169<br>d [+0.151, +0.187]<br>c [+0.139, +0.198] | +0.505<br>d [+0.487, +0.522]<br>c [+0.474, +0.534] | 125.3 | 36.4 |
| G causal GP | - | +0.405<br>d [+0.384, +0.425]<br>c [+0.362, +0.441] | 137.4 | 38.3 |
| H hold | -0.680<br>d [-0.738, -0.623]<br>c [-0.788, -0.567] | - | 178.1 | 45.5 |
| M5 median of last five | -0.200<br>d [-0.237, -0.162]<br>c [-0.260, -0.134] | +0.286<br>d [+0.249, +0.321]<br>c [+0.209, +0.352] | 150.5 | 40.4 |
| network (first stage), as built | +0.062<br>d [+0.029, +0.092]<br>c [+0.016, +0.108] | +0.442<br>d [+0.416, +0.465]<br>c [+0.400, +0.480] | 133.1 | 41.3 |
| history-only twin, as built | +0.028<br>d [-0.002, +0.058]<br>c [-0.022, +0.078] | +0.422<br>d [+0.396, +0.446]<br>c [+0.373, +0.467] | 135.5 | 39.5 |
| two-stage estimator, as built | +0.134<br>d [+0.111, +0.156]<br>c [+0.096, +0.171] | +0.484<br>d [+0.465, +0.502]<br>c [+0.454, +0.513] | 127.9 | 38.0 |
| two-stage twin, as built | +0.099<br>d [+0.077, +0.122]<br>c [+0.063, +0.136] | +0.464<br>d [+0.441, +0.485]<br>c [+0.421, +0.503] | 130.4 | 37.6 |
| skill: TX over c2 gain-fitted | +0.033<br>d [+0.016, +0.050]<br>c [+0.007, +0.060] | | | |
| skill: TX over c2 as built | +0.073<br>d [+0.053, +0.093]<br>c [+0.044, +0.099] | | | |
| skill: T0 over t2 as built | +0.030<br>d [+0.014, +0.046]<br>c [+0.006, +0.051] | | | |

### V1 random (r2-r4), target L5

| estimator | skill over G | skill over H | RMSE (eV) | median abs. error (eV) |
|---|---|---|---|---|
| T0[L5] | +0.166<br>d [+0.146, +0.187]<br>c [+0.133, +0.201] | +0.534<br>d [+0.509, +0.558]<br>c [+0.483, +0.580] | 132.9 | 38.7 |
| TX[L5] | +0.229<br>d [+0.199, +0.253]<br>c [+0.194, +0.264] | +0.569<br>d [+0.544, +0.592]<br>c [+0.528, +0.608] | 127.8 | 37.7 |
| TB[L5] | +0.221<br>d [+0.195, +0.244]<br>c [+0.188, +0.254] | +0.565<br>d [+0.541, +0.587]<br>c [+0.522, +0.604] | 128.5 | 37.8 |
| TE[L5] | +0.193<br>d [+0.174, +0.213]<br>c [+0.162, +0.225] | +0.549<br>d [+0.525, +0.572]<br>c [+0.502, +0.593] | 130.7 | 38.4 |
| TXlag[L5] | +0.194<br>d [+0.168, +0.217]<br>c [+0.163, +0.226] | +0.549<br>d [+0.524, +0.573]<br>c [+0.503, +0.592] | 130.7 | 38.3 |
| K0[L5] | +0.003<br>d [-0.028, +0.035]<br>c [-0.055, +0.070] | +0.443<br>d [+0.408, +0.476]<br>c [+0.363, +0.515] | 145.3 | 39.8 |
| KX[L5] | +0.058<br>d [+0.030, +0.085]<br>c [+0.008, +0.113] | +0.473<br>d [+0.443, +0.503]<br>c [+0.405, +0.535] | 141.3 | 39.2 |
| network, gain fitted [L5] | +0.114<br>d [+0.090, +0.140]<br>c [+0.076, +0.154] | +0.505<br>d [+0.479, +0.530]<br>c [+0.455, +0.552] | 137.0 | 41.6 |
| two-stage, gain fitted [L5] | +0.176<br>d [+0.157, +0.195]<br>c [+0.146, +0.207] | +0.539<br>d [+0.517, +0.561]<br>c [+0.493, +0.581] | 132.2 | 39.5 |
| G causal GP | - | +0.441<br>d [+0.417, +0.464]<br>c [+0.389, +0.486] | 145.6 | 40.7 |
| H hold | -0.789<br>d [-0.865, -0.716]<br>c [-0.945, -0.636] | - | 194.7 | 47.0 |
| M5 median of last five | -0.125<br>d [-0.158, -0.089]<br>c [-0.184, -0.060] | +0.372<br>d [+0.334, +0.409]<br>c [+0.290, +0.445] | 154.4 | 40.6 |
| network (first stage), as built | +0.070<br>d [+0.038, +0.100]<br>c [+0.030, +0.114] | +0.480<br>d [+0.452, +0.507]<br>c [+0.428, +0.528] | 140.4 | 45.0 |
| history-only twin, as built | +0.055<br>d [+0.025, +0.087]<br>c [+0.006, +0.107] | +0.472<br>d [+0.443, +0.500]<br>c [+0.413, +0.527] | 141.5 | 42.2 |
| two-stage estimator, as built | +0.133<br>d [+0.109, +0.155]<br>c [+0.097, +0.168] | +0.515<br>d [+0.492, +0.537]<br>c [+0.471, +0.557] | 135.6 | 41.4 |
| two-stage twin, as built | +0.121<br>d [+0.098, +0.146]<br>c [+0.084, +0.163] | +0.509<br>d [+0.482, +0.535]<br>c [+0.454, +0.558] | 136.5 | 40.3 |
| skill: TX over c2 gain-fitted | +0.065<br>d [+0.034, +0.087]<br>c [+0.031, +0.095] | | | |
| skill: TX over c2 as built | +0.112<br>d [+0.079, +0.139]<br>c [+0.079, +0.140] | | | |
| skill: T0 over t2 as built | +0.051<br>d [+0.033, +0.068]<br>c [+0.027, +0.069] | | | |

### V1 random (r2-r4), target L10

| estimator | skill over G | skill over H | RMSE (eV) | median abs. error (eV) |
|---|---|---|---|---|
| T0[L10] | +0.213<br>d [+0.190, +0.237]<br>c [+0.172, +0.255] | +0.577<br>d [+0.551, +0.604]<br>c [+0.523, +0.627] | 131.1 | 38.9 |
| TX[L10] | +0.271<br>d [+0.235, +0.299]<br>c [+0.230, +0.311] | +0.608<br>d [+0.580, +0.634]<br>c [+0.560, +0.653] | 126.1 | 38.2 |
| network, gain fitted [L10] | +0.118<br>d [+0.090, +0.147]<br>c [+0.074, +0.167] | +0.526<br>d [+0.499, +0.553]<br>c [+0.469, +0.581] | 138.8 | 42.7 |
| two-stage, gain fitted [L10] | +0.170<br>d [+0.148, +0.192]<br>c [+0.134, +0.210] | +0.554<br>d [+0.529, +0.579]<br>c [+0.499, +0.605] | 134.7 | 40.4 |
| G causal GP | - | +0.463<br>d [+0.437, +0.488]<br>c [+0.407, +0.512] | 147.8 | 40.7 |
| H hold | -0.861<br>d [-0.953, -0.776]<br>c [-1.049, -0.685] | - | 201.6 | 49.3 |
| M5 median of last five | -0.103<br>d [-0.137, -0.068]<br>c [-0.161, -0.035] | +0.407<br>d [+0.370, +0.445]<br>c [+0.323, +0.484] | 155.2 | 40.9 |
| network (first stage), as built | +0.069<br>d [+0.036, +0.102]<br>c [+0.026, +0.121] | +0.500<br>d [+0.471, +0.529]<br>c [+0.439, +0.557] | 142.5 | 46.6 |
| history-only twin, as built | +0.066<br>d [+0.033, +0.100]<br>c [+0.010, +0.124] | +0.498<br>d [+0.469, +0.528]<br>c [+0.436, +0.558] | 142.8 | 43.0 |
| two-stage estimator, as built | +0.115<br>d [+0.088, +0.142]<br>c [+0.074, +0.161] | +0.525<br>d [+0.499, +0.550]<br>c [+0.469, +0.576] | 139.0 | 42.9 |
| two-stage twin, as built | +0.128<br>d [+0.104, +0.154]<br>c [+0.086, +0.175] | +0.532<br>d [+0.504, +0.559]<br>c [+0.472, +0.587] | 138.0 | 41.0 |
| skill: TX over c2 gain-fitted | +0.122<br>d [+0.083, +0.151]<br>c [+0.079, +0.160] | | | |
| skill: TX over c2 as built | +0.176<br>d [+0.136, +0.208]<br>c [+0.133, +0.215] | | | |
| skill: T0 over t2 as built | +0.097<br>d [+0.076, +0.117]<br>c [+0.068, +0.123] | | | |

### V2 cluster-blocked (3 assignments), target F

| estimator | skill over G | skill over H | RMSE (eV) | median abs. error (eV) |
|---|---|---|---|---|
| T0[F] | +0.093<br>d [+0.083, +0.103]<br>c [+0.081, +0.106] | +0.377<br>d [+0.364, +0.390]<br>c [+0.355, +0.394] | 183.6 | 41.8 |
| TX[F] | +0.158<br>d [+0.140, +0.176]<br>c [+0.133, +0.188] | +0.422<br>d [+0.406, +0.437]<br>c [+0.399, +0.444] | 176.8 | 40.2 |
| K0[F] | +0.003<br>d [-0.010, +0.016]<br>c [-0.015, +0.020] | +0.316<br>d [+0.300, +0.331]<br>c [+0.282, +0.340] | 192.4 | 43.3 |
| KX[F] | +0.051<br>d [+0.037, +0.065]<br>c [+0.031, +0.070] | +0.349<br>d [+0.335, +0.362]<br>c [+0.321, +0.368] | 187.7 | 41.9 |
| G causal GP | - | +0.314<br>d [+0.298, +0.329]<br>c [+0.285, +0.335] | 192.7 | 44.1 |
| H hold | -0.457<br>d [-0.490, -0.424]<br>c [-0.505, -0.398] | - | 232.6 | 45.4 |
| M5 median of last five | -0.104<br>d [-0.125, -0.084]<br>c [-0.143, -0.073] | +0.242<br>d [+0.218, +0.265]<br>c [+0.188, +0.280] | 202.5 | 44.4 |

### V2 cluster-blocked (3 assignments), target L5

| estimator | skill over G | skill over H | RMSE (eV) | median abs. error (eV) |
|---|---|---|---|---|
| T0[L5] | +0.150<br>d [+0.128, +0.173]<br>c [+0.111, +0.191] | +0.525<br>d [+0.499, +0.550]<br>c [+0.472, +0.575] | 134.2 | 38.9 |
| TX[L5] | +0.199<br>d [+0.176, +0.222]<br>c [+0.165, +0.236] | +0.552<br>d [+0.529, +0.576]<br>c [+0.505, +0.596] | 130.3 | 38.3 |
| K0[L5] | -0.001<br>d [-0.033, +0.031]<br>c [-0.062, +0.068] | +0.440<br>d [+0.405, +0.474]<br>c [+0.360, +0.514] | 145.7 | 39.7 |
| KX[L5] | +0.053<br>d [+0.025, +0.082]<br>c [+0.001, +0.110] | +0.471<br>d [+0.440, +0.501]<br>c [+0.401, +0.534] | 141.6 | 39.2 |
| G causal GP | - | +0.441<br>d [+0.417, +0.464]<br>c [+0.389, +0.486] | 145.6 | 40.7 |
| H hold | -0.789<br>d [-0.865, -0.716]<br>c [-0.944, -0.637] | - | 194.7 | 47.0 |
| M5 median of last five | -0.125<br>d [-0.159, -0.089]<br>c [-0.182, -0.060] | +0.372<br>d [+0.333, +0.409]<br>c [+0.290, +0.445] | 154.4 | 40.6 |

### V3 rolling origin (fifths 2-5 pooled), target F

| estimator | skill over G | skill over H | RMSE (eV) | median abs. error (eV) |
|---|---|---|---|---|
| T0[F] | +0.074<br>d [+0.064, +0.084]<br>c [+0.064, +0.085] | +0.365<br>d [+0.350, +0.381]<br>c [+0.337, +0.386] | 206.1 | 49.0 |
| TX[F] | +0.116<br>d [+0.101, +0.131]<br>c [+0.093, +0.143] | +0.394<br>d [+0.377, +0.410]<br>c [+0.363, +0.419] | 201.3 | 47.7 |
| K0[F] | +0.005<br>d [-0.009, +0.020]<br>c [-0.012, +0.023] | +0.318<br>d [+0.302, +0.334]<br>c [+0.286, +0.341] | 213.5 | 49.9 |
| KX[F] | +0.046<br>d [+0.030, +0.062]<br>c [+0.026, +0.066] | +0.346<br>d [+0.331, +0.360]<br>c [+0.318, +0.366] | 209.1 | 48.3 |
| G causal GP | - | +0.315<br>d [+0.297, +0.331]<br>c [+0.284, +0.337] | 214.1 | 51.5 |
| H hold | -0.459<br>d [-0.495, -0.423]<br>c [-0.508, -0.396] | - | 258.7 | 52.5 |
| M5 median of last five | -0.096<br>d [-0.118, -0.075]<br>c [-0.138, -0.063] | +0.249<br>d [+0.223, +0.273]<br>c [+0.191, +0.288] | 224.2 | 50.3 |

### V3 rolling origin (fifths 2-5 pooled), target L5

| estimator | skill over G | skill over H | RMSE (eV) | median abs. error (eV) |
|---|---|---|---|---|
| T0[L5] | +0.134<br>d [+0.112, +0.156]<br>c [+0.094, +0.178] | +0.525<br>d [+0.497, +0.553]<br>c [+0.465, +0.581] | 148.5 | 44.6 |
| TX[L5] | +0.161<br>d [+0.138, +0.184]<br>c [+0.120, +0.208] | +0.540<br>d [+0.514, +0.566]<br>c [+0.485, +0.592] | 146.2 | 44.3 |
| K0[L5] | +0.007<br>d [-0.029, +0.044]<br>c [-0.059, +0.085] | +0.455<br>d [+0.418, +0.492]<br>c [+0.371, +0.533] | 159.1 | 44.6 |
| KX[L5] | +0.049<br>d [+0.017, +0.081]<br>c [-0.008, +0.114] | +0.478<br>d [+0.445, +0.511]<br>c [+0.404, +0.547] | 155.7 | 44.3 |
| G causal GP | - | +0.452<br>d [+0.426, +0.476]<br>c [+0.398, +0.497] | 159.6 | 46.2 |
| H hold | -0.824<br>d [-0.910, -0.742]<br>c [-0.989, -0.660] | - | 215.6 | 53.0 |
| M5 median of last five | -0.105<br>d [-0.143, -0.065]<br>c [-0.170, -0.033] | +0.394<br>d [+0.352, +0.435]<br>c [+0.307, +0.472] | 167.8 | 44.5 |

### V4 chronological: fit to 31991, score from 32312, target F

| estimator | skill over G | skill over H | RMSE (eV) | median abs. error (eV) |
|---|---|---|---|---|
| T0[F] | +0.068<br>d [+0.046, +0.091]<br>c [+0.047, +0.096] | +0.346<br>d [+0.310, +0.381]<br>c [+0.294, +0.395] | 248.5 | 57.5 |
| TX[F] | +0.086<br>d [+0.063, +0.111]<br>c [+0.056, +0.125] | +0.359<br>d [+0.323, +0.393]<br>c [+0.299, +0.415] | 246.1 | 53.9 |
| TB[F] | +0.087<br>d [+0.065, +0.110]<br>c [+0.063, +0.117] | +0.360<br>d [+0.327, +0.392]<br>c [+0.308, +0.408] | 245.9 | 55.7 |
| TE[F] | +0.086<br>d [+0.066, +0.108]<br>c [+0.061, +0.122] | +0.359<br>d [+0.328, +0.389]<br>c [+0.307, +0.410] | 246.1 | 54.5 |
| TXlag[F] | +0.068<br>d [+0.048, +0.090]<br>c [+0.039, +0.106] | +0.346<br>d [+0.310, +0.381]<br>c [+0.287, +0.402] | 248.5 | 55.1 |
| K0[F] | +0.058<br>d [+0.019, +0.089]<br>c [+0.011, +0.097] | +0.339<br>d [+0.313, +0.365]<br>c [+0.315, +0.359] | 249.8 | 52.5 |
| KX[F] | +0.083<br>d [+0.042, +0.116]<br>c [+0.043, +0.118] | +0.357<br>d [+0.330, +0.384]<br>c [+0.327, +0.379] | 246.5 | 51.5 |
| G causal GP | - | +0.298<br>d [+0.268, +0.329]<br>c [+0.254, +0.337] | 257.4 | 62.2 |
| H hold | -0.425<br>d [-0.489, -0.366]<br>c [-0.508, -0.340] | - | 307.3 | 56.3 |
| M5 median of last five | +0.009<br>d [-0.034, +0.047]<br>c [-0.028, +0.048] | +0.305<br>d [+0.268, +0.340]<br>c [+0.268, +0.337] | 256.2 | 49.7 |

### V4 chronological: fit to 31991, score from 32312, target L5

| estimator | skill over G | skill over H | RMSE (eV) | median abs. error (eV) |
|---|---|---|---|---|
| T0[L5] | +0.254<br>d [+0.208, +0.301]<br>c [+0.214, +0.307] | +0.665<br>d [+0.621, +0.710]<br>c [+0.635, +0.701] | 143.9 | 47.3 |
| TX[L5] | +0.317<br>d [+0.262, +0.371]<br>c [+0.257, +0.381] | +0.693<br>d [+0.649, +0.737]<br>c [+0.663, +0.725] | 137.7 | 44.6 |
| TB[L5] | +0.301<br>d [+0.247, +0.353]<br>c [+0.250, +0.360] | +0.686<br>d [+0.642, +0.729]<br>c [+0.658, +0.719] | 139.3 | 45.5 |
| TE[L5] | +0.297<br>d [+0.245, +0.350]<br>c [+0.247, +0.352] | +0.685<br>d [+0.640, +0.729]<br>c [+0.655, +0.715] | 139.6 | 45.9 |
| TXlag[L5] | +0.284<br>d [+0.230, +0.339]<br>c [+0.227, +0.348] | +0.679<br>d [+0.634, +0.723]<br>c [+0.650, +0.708] | 141.0 | 45.8 |
| K0[L5] | +0.302<br>d [+0.215, +0.387]<br>c [+0.170, +0.404] | +0.687<br>d [+0.629, +0.743]<br>c [+0.628, +0.734] | 139.1 | 42.8 |
| KX[L5] | +0.297<br>d [+0.211, +0.381]<br>c [+0.170, +0.395] | +0.685<br>d [+0.627, +0.740]<br>c [+0.628, +0.729] | 139.7 | 42.3 |
| G causal GP | - | +0.551<br>d [+0.508, +0.595]<br>c [+0.505, +0.595] | 166.6 | 51.9 |
| H hold | -1.228<br>d [-1.469, -1.034]<br>c [-1.471, -1.021] | - | 248.7 | 51.7 |
| M5 median of last five | +0.194<br>d [+0.076, +0.317]<br>c [+0.099, +0.298] | +0.638<br>d [+0.558, +0.716]<br>c [+0.590, +0.694] | 149.5 | 40.9 |

### V4 chronological: fit to 32310, score from 32312, target F

| estimator | skill over G | skill over H | RMSE (eV) | median abs. error (eV) |
|---|---|---|---|---|
| T0[F] | +0.085<br>d [+0.066, +0.106]<br>c [+0.066, +0.110] | +0.358<br>d [+0.327, +0.388]<br>c [+0.313, +0.403] | 246.3 | 58.5 |
| TX[F] | +0.114<br>d [+0.089, +0.137]<br>c [+0.084, +0.144] | +0.378<br>d [+0.350, +0.405]<br>c [+0.334, +0.423] | 242.4 | 53.3 |
| TB[F] | +0.118<br>d [+0.094, +0.140]<br>c [+0.090, +0.144] | +0.381<br>d [+0.353, +0.408]<br>c [+0.342, +0.421] | 241.8 | 55.1 |
| TE[F] | +0.088<br>d [+0.067, +0.112]<br>c [+0.061, +0.123] | +0.360<br>d [+0.332, +0.388]<br>c [+0.313, +0.410] | 245.8 | 54.2 |
| TXlag[F] | +0.094<br>d [+0.075, +0.114]<br>c [+0.068, +0.127] | +0.364<br>d [+0.336, +0.393]<br>c [+0.319, +0.412] | 245.0 | 55.3 |
| K0[F] | +0.060<br>d [+0.021, +0.090]<br>c [+0.014, +0.097] | +0.340<br>d [+0.314, +0.365]<br>c [+0.315, +0.360] | 249.6 | 52.5 |
| KX[F] | +0.085<br>d [+0.044, +0.117]<br>c [+0.044, +0.117] | +0.358<br>d [+0.330, +0.384]<br>c [+0.328, +0.381] | 246.3 | 51.5 |
| G causal GP | - | +0.298<br>d [+0.268, +0.328]<br>c [+0.255, +0.337] | 257.4 | 62.2 |
| H hold | -0.425<br>d [-0.489, -0.366]<br>c [-0.509, -0.342] | - | 307.3 | 56.3 |
| M5 median of last five | +0.009<br>d [-0.034, +0.048]<br>c [-0.028, +0.047] | +0.305<br>d [+0.268, +0.340]<br>c [+0.268, +0.337] | 256.2 | 49.7 |

### V4 chronological: fit to 32310, score from 32312, target L5

| estimator | skill over G | skill over H | RMSE (eV) | median abs. error (eV) |
|---|---|---|---|---|
| T0[L5] | +0.312<br>d [+0.258, +0.365]<br>c [+0.251, +0.378] | +0.691<br>d [+0.648, +0.734]<br>c [+0.664, +0.723] | 138.2 | 45.5 |
| TX[L5] | +0.361<br>d [+0.308, +0.413]<br>c [+0.298, +0.425] | +0.713<br>d [+0.672, +0.753]<br>c [+0.685, +0.743] | 133.2 | 42.9 |
| TB[L5] | +0.348<br>d [+0.293, +0.402]<br>c [+0.284, +0.415] | +0.707<br>d [+0.665, +0.749]<br>c [+0.679, +0.738] | 134.5 | 43.2 |
| TE[L5] | +0.345<br>d [+0.298, +0.392]<br>c [+0.290, +0.403] | +0.706<br>d [+0.666, +0.745]<br>c [+0.679, +0.737] | 134.8 | 44.5 |
| TXlag[L5] | +0.341<br>d [+0.287, +0.395]<br>c [+0.273, +0.409] | +0.704<br>d [+0.662, +0.745]<br>c [+0.674, +0.736] | 135.2 | 43.8 |
| K0[L5] | +0.309<br>d [+0.220, +0.392]<br>c [+0.178, +0.408] | +0.690<br>d [+0.631, +0.745]<br>c [+0.632, +0.736] | 138.5 | 42.6 |
| KX[L5] | +0.305<br>d [+0.217, +0.388]<br>c [+0.181, +0.401] | +0.688<br>d [+0.630, +0.743]<br>c [+0.633, +0.732] | 138.9 | 42.1 |
| G causal GP | - | +0.551<br>d [+0.508, +0.595]<br>c [+0.506, +0.595] | 166.6 | 51.9 |
| H hold | -1.228<br>d [-1.471, -1.033]<br>c [-1.472, -1.023] | - | 248.7 | 51.7 |
| M5 median of last five | +0.194<br>d [+0.076, +0.313]<br>c [+0.101, +0.298] | +0.638<br>d [+0.559, +0.715]<br>c [+0.591, +0.695] | 149.5 | 40.9 |

## V1 by chronological block and by fifth of the BES level of the row

Strata by a predictor or by shot number only. No stratum is defined by the scored value or by later values. BES level = raw mean of the nine BES channels at row t; quintile edges 0.111, 0.17, 0.199, 0.235.

### Target F

| stratum | rows / discharges / clusters | trees TX over T0 | filters KX over K0 | TB over T0 (BES only) | TE over T0 (ECEI only) | TXlag over T0 (one row old) | network over twin, as built | two-stage over two-stage twin, as built | network over twin, gain fitted | two-stage over two-stage twin, gain fitted | T0 over G |
|---|---|---|---|---|---|---|---|---|---|---|---|
| block up to 31991 | 168,095 / 412 / 54 | +0.130 (+13.0 %)<br>d [+0.109, +0.151]<br>c [+0.091, +0.176] | +0.070 (+7.0 %)<br>d [+0.058, +0.084]<br>c [+0.043, +0.099] | +0.088 (+8.8 %)<br>d [+0.074, +0.102]<br>c [+0.062, +0.116] | +0.092 (+9.2 %)<br>d [+0.071, +0.113]<br>c [+0.053, +0.137] | +0.025 (+2.5 %)<br>d [+0.017, +0.032]<br>c [+0.015, +0.034] | +0.079 (+7.9 %)<br>d [+0.040, +0.115]<br>c [+0.018, +0.126] | +0.145 (+14.5 %)<br>d [+0.115, +0.173]<br>c [+0.094, +0.196] | +0.091 (+9.1 %)<br>d [+0.067, +0.116]<br>c [+0.052, +0.126] | +0.146 (+14.6 %)<br>d [+0.120, +0.171]<br>c [+0.100, +0.191] | +0.109 (+10.9 %)<br>d [+0.095, +0.123]<br>c [+0.090, +0.131] |
| block 32002-32310 | 34,515 / 128 / 14 | +0.038 (+3.8 %)<br>d [+0.020, +0.055]<br>c [+0.017, +0.062] | +0.024 (+2.4 %)<br>d [+0.011, +0.038]<br>c [+0.006, +0.046] | +0.038 (+3.8 %)<br>d [+0.024, +0.051]<br>c [+0.025, +0.058] | +0.009 (+0.9 %)<br>d [-0.004, +0.022]<br>c [-0.008, +0.028] | +0.003 (+0.3 %)<br>d [-0.005, +0.011]<br>c [-0.008, +0.017] | +0.016 (+1.6 %)<br>d [-0.017, +0.044]<br>c [-0.015, +0.039] | +0.037 (+3.7 %)<br>d [+0.008, +0.061]<br>c [+0.003, +0.065] | +0.033 (+3.3 %)<br>d [+0.013, +0.052]<br>c [+0.017, +0.048] | +0.043 (+4.3 %)<br>d [+0.019, +0.063]<br>c [+0.015, +0.066] | +0.085 (+8.5 %)<br>d [+0.068, +0.101]<br>c [+0.071, +0.095] |
| block from 32312 | 20,062 / 95 / 19 | +0.043 (+4.3 %)<br>d [+0.023, +0.064]<br>c [+0.021, +0.065] | +0.027 (+2.7 %)<br>d [+0.016, +0.039]<br>c [+0.006, +0.051] | +0.034 (+3.4 %)<br>d [+0.015, +0.051]<br>c [+0.013, +0.050] | +0.013 (+1.3 %)<br>d [-0.002, +0.029]<br>c [-0.004, +0.029] | +0.009 (+0.9 %)<br>d [-0.001, +0.019]<br>c [-0.006, +0.026] | +0.031 (+3.1 %)<br>d [+0.013, +0.050]<br>c [+0.004, +0.055] | +0.052 (+5.2 %)<br>d [+0.030, +0.075]<br>c [+0.024, +0.078] | +0.033 (+3.3 %)<br>d [+0.020, +0.046]<br>c [+0.012, +0.053] | +0.053 (+5.3 %)<br>d [+0.033, +0.074]<br>c [+0.028, +0.077] | +0.094 (+9.4 %)<br>d [+0.076, +0.112]<br>c [+0.072, +0.115] |
| BES level fifth 1 | 44,535 / 485 / 84 | +0.039 (+3.9 %)<br>d [+0.024, +0.053]<br>c [+0.019, +0.059] | +0.009 (+0.9 %)<br>d [+0.002, +0.017]<br>c [-0.002, +0.020] | +0.037 (+3.7 %)<br>d [+0.024, +0.050]<br>c [+0.020, +0.054] | +0.014 (+1.4 %)<br>d [+0.003, +0.025]<br>c [+0.002, +0.029] | +0.005 (+0.5 %)<br>d [-0.005, +0.014]<br>c [-0.006, +0.017] | -0.003 (-0.3 %)<br>d [-0.044, +0.018]<br>c [-0.040, +0.019] | +0.033 (+3.3 %)<br>d [+0.002, +0.055]<br>c [+0.001, +0.061] | +0.013 (+1.3 %)<br>d [-0.013, +0.028]<br>c [-0.009, +0.029] | +0.038 (+3.8 %)<br>d [+0.012, +0.056]<br>c [+0.011, +0.062] | +0.089 (+8.9 %)<br>d [+0.070, +0.108]<br>c [+0.066, +0.113] |
| BES level fifth 2 | 44,534 / 518 / 76 | +0.110 (+11.0 %)<br>d [+0.085, +0.139]<br>c [+0.063, +0.177] | +0.066 (+6.6 %)<br>d [+0.051, +0.082]<br>c [+0.039, +0.098] | +0.081 (+8.1 %)<br>d [+0.065, +0.098]<br>c [+0.053, +0.118] | +0.071 (+7.1 %)<br>d [+0.046, +0.099]<br>c [+0.025, +0.136] | +0.018 (+1.8 %)<br>d [+0.009, +0.029]<br>c [+0.007, +0.032] | +0.075 (+7.5 %)<br>d [+0.007, +0.136]<br>c [-0.018, +0.154] | +0.124 (+12.4 %)<br>d [+0.078, +0.171]<br>c [+0.053, +0.204] | +0.082 (+8.2 %)<br>d [+0.041, +0.127]<br>c [+0.025, +0.141] | +0.125 (+12.5 %)<br>d [+0.085, +0.167]<br>c [+0.062, +0.198] | +0.122 (+12.2 %)<br>d [+0.103, +0.142]<br>c [+0.099, +0.151] |
| BES level fifth 3 | 44,534 / 438 / 60 | +0.141 (+14.1 %)<br>d [+0.114, +0.167]<br>c [+0.094, +0.188] | +0.102 (+10.2 %)<br>d [+0.089, +0.117]<br>c [+0.080, +0.127] | +0.092 (+9.2 %)<br>d [+0.074, +0.110]<br>c [+0.063, +0.119] | +0.091 (+9.1 %)<br>d [+0.064, +0.117]<br>c [+0.040, +0.142] | +0.032 (+3.2 %)<br>d [+0.023, +0.041]<br>c [+0.021, +0.043] | +0.120 (+12.0 %)<br>d [+0.090, +0.148]<br>c [+0.073, +0.162] | +0.175 (+17.5 %)<br>d [+0.146, +0.205]<br>c [+0.128, +0.220] | +0.121 (+12.1 %)<br>d [+0.100, +0.141]<br>c [+0.088, +0.151] | +0.172 (+17.2 %)<br>d [+0.146, +0.199]<br>c [+0.130, +0.213] | +0.103 (+10.3 %)<br>d [+0.080, +0.130]<br>c [+0.075, +0.144] |
| BES level fifth 4 | 44,534 / 379 / 56 | +0.123 (+12.3 %)<br>d [+0.101, +0.145]<br>c [+0.096, +0.152] | +0.098 (+9.8 %)<br>d [+0.077, +0.120]<br>c [+0.066, +0.136] | +0.089 (+8.9 %)<br>d [+0.075, +0.104]<br>c [+0.069, +0.110] | +0.071 (+7.1 %)<br>d [+0.050, +0.093]<br>c [+0.044, +0.099] | +0.028 (+2.8 %)<br>d [+0.017, +0.039]<br>c [+0.016, +0.040] | +0.115 (+11.5 %)<br>d [+0.083, +0.150]<br>c [+0.072, +0.164] | +0.146 (+14.6 %)<br>d [+0.118, +0.174]<br>c [+0.112, +0.184] | +0.128 (+12.8 %)<br>d [+0.105, +0.154]<br>c [+0.092, +0.170] | +0.149 (+14.9 %)<br>d [+0.124, +0.175]<br>c [+0.118, +0.186] | +0.093 (+9.3 %)<br>d [+0.068, +0.117]<br>c [+0.055, +0.125] |
| BES level fifth 5 | 44,535 / 317 / 51 | +0.085 (+8.5 %)<br>d [+0.061, +0.110]<br>c [+0.060, +0.123] | +0.023 (+2.3 %)<br>d [+0.010, +0.038]<br>c [+0.004, +0.069] | +0.045 (+4.5 %)<br>d [+0.030, +0.064]<br>c [+0.029, +0.090] | +0.066 (+6.6 %)<br>d [+0.044, +0.089]<br>c [+0.031, +0.089] | +0.015 (+1.5 %)<br>d [+0.003, +0.024]<br>c [-0.009, +0.025] | +0.025 (+2.5 %)<br>d [-0.002, +0.053]<br>c [-0.009, +0.072] | +0.086 (+8.6 %)<br>d [+0.061, +0.113]<br>c [+0.054, +0.125] | +0.050 (+5.0 %)<br>d [+0.031, +0.072]<br>c [+0.026, +0.103] | +0.090 (+9.0 %)<br>d [+0.067, +0.114]<br>c [+0.062, +0.129] | +0.057 (+5.7 %)<br>d [+0.040, +0.078]<br>c [+0.037, +0.092] |

### Target N1

| stratum | rows / discharges / clusters | trees TX over T0 | network over twin, as built | two-stage over two-stage twin, as built | network over twin, gain fitted | two-stage over two-stage twin, gain fitted | T0 over G |
|---|---|---|---|---|---|---|---|
| block up to 31991 | 163,642 / 404 / 53 | +0.061 (+6.1 %)<br>d [+0.048, +0.074]<br>c [+0.038, +0.087] | +0.034 (+3.4 %)<br>d [+0.001, +0.064]<br>c [-0.016, +0.073] | +0.044 (+4.4 %)<br>d [+0.020, +0.068]<br>c [+0.007, +0.086] | +0.060 (+6.0 %)<br>d [+0.044, +0.078]<br>c [+0.035, +0.087] | +0.067 (+6.7 %)<br>d [+0.053, +0.083]<br>c [+0.043, +0.095] | +0.077 (+7.7 %)<br>d [+0.063, +0.091]<br>c [+0.054, +0.102] |
| block 32002-32310 | 31,837 / 128 / 14 | +0.018 (+1.8 %)<br>d [+0.003, +0.033]<br>c [+0.002, +0.041] | +0.001 (+0.1 %)<br>d [-0.051, +0.034]<br>c [-0.051, +0.034] | +0.005 (+0.5 %)<br>d [-0.026, +0.031]<br>c [-0.028, +0.034] | +0.027 (+2.7 %)<br>d [+0.004, +0.045]<br>c [+0.009, +0.048] | +0.026 (+2.6 %)<br>d [+0.012, +0.041]<br>c [+0.013, +0.045] | +0.061 (+6.1 %)<br>d [+0.041, +0.080]<br>c [+0.035, +0.073] |
| block from 32312 | 19,289 / 95 / 19 | +0.021 (+2.1 %)<br>d [+0.008, +0.035]<br>c [+0.000, +0.046] | +0.009 (+0.9 %)<br>d [-0.009, +0.028]<br>c [-0.025, +0.034] | +0.013 (+1.3 %)<br>d [-0.009, +0.032]<br>c [-0.013, +0.036] | +0.018 (+1.8 %)<br>d [+0.007, +0.030]<br>c [-0.002, +0.036] | +0.023 (+2.3 %)<br>d [+0.010, +0.037]<br>c [+0.005, +0.041] | +0.112 (+11.2 %)<br>d [+0.088, +0.136]<br>c [+0.082, +0.144] |
| BES level fifth 1 | 41,951 / 462 / 81 | +0.021 (+2.1 %)<br>d [+0.006, +0.038]<br>c [+0.003, +0.044] | -0.012 (-1.2 %)<br>d [-0.059, +0.013]<br>c [-0.061, +0.016] | +0.005 (+0.5 %)<br>d [-0.025, +0.029]<br>c [-0.032, +0.039] | +0.013 (+1.3 %)<br>d [-0.009, +0.027]<br>c [-0.008, +0.029] | +0.023 (+2.3 %)<br>d [+0.008, +0.037]<br>c [+0.004, +0.044] | +0.104 (+10.4 %)<br>d [+0.082, +0.125]<br>c [+0.080, +0.132] |
| BES level fifth 2 | 41,826 / 510 / 74 | +0.046 (+4.6 %)<br>d [+0.032, +0.062]<br>c [+0.025, +0.075] | +0.049 (+4.9 %)<br>d [-0.001, +0.092]<br>c [-0.020, +0.099] | +0.043 (+4.3 %)<br>d [+0.008, +0.076]<br>c [-0.001, +0.091] | +0.061 (+6.1 %)<br>d [+0.037, +0.087]<br>c [+0.028, +0.094] | +0.061 (+6.1 %)<br>d [+0.041, +0.083]<br>c [+0.036, +0.095] | +0.077 (+7.7 %)<br>d [+0.058, +0.096]<br>c [+0.053, +0.102] |
| BES level fifth 3 | 43,621 / 427 / 58 | +0.059 (+5.9 %)<br>d [+0.045, +0.073]<br>c [+0.036, +0.088] | +0.050 (+5.0 %)<br>d [+0.025, +0.074]<br>c [+0.018, +0.087] | +0.049 (+4.9 %)<br>d [+0.024, +0.072]<br>c [+0.010, +0.093] | +0.067 (+6.7 %)<br>d [+0.052, +0.081]<br>c [+0.047, +0.092] | +0.067 (+6.7 %)<br>d [+0.052, +0.082]<br>c [+0.043, +0.095] | +0.057 (+5.7 %)<br>d [+0.040, +0.076]<br>c [+0.036, +0.088] |
| BES level fifth 4 | 43,968 / 367 / 55 | +0.059 (+5.9 %)<br>d [+0.040, +0.080]<br>c [+0.029, +0.092] | +0.025 (+2.5 %)<br>d [-0.008, +0.063]<br>c [-0.025, +0.079] | +0.043 (+4.3 %)<br>d [+0.017, +0.071]<br>c [-0.002, +0.089] | +0.066 (+6.6 %)<br>d [+0.043, +0.091]<br>c [+0.029, +0.105] | +0.070 (+7.0 %)<br>d [+0.052, +0.090]<br>c [+0.038, +0.104] | +0.076 (+7.6 %)<br>d [+0.061, +0.092]<br>c [+0.054, +0.097] |
| BES level fifth 5 | 43,402 / 307 / 48 | +0.050 (+5.0 %)<br>d [+0.033, +0.069]<br>c [+0.022, +0.083] | -0.033 (-3.3 %)<br>d [-0.070, -0.005]<br>c [-0.134, +0.002] | -0.003 (-0.3 %)<br>d [-0.030, +0.021]<br>c [-0.060, +0.031] | +0.022 (+2.2 %)<br>d [+0.006, +0.037]<br>c [-0.009, +0.044] | +0.031 (+3.1 %)<br>d [+0.016, +0.046]<br>c [+0.006, +0.060] | +0.041 (+4.1 %)<br>d [+0.016, +0.069]<br>c [+0.012, +0.099] |

### Target L3

| stratum | rows / discharges / clusters | trees TX over T0 | network over twin, as built | two-stage over two-stage twin, as built | network over twin, gain fitted | two-stage over two-stage twin, gain fitted | T0 over G |
|---|---|---|---|---|---|---|---|
| block up to 31991 | 157,310 / 401 / 53 | +0.106 (+10.6 %)<br>d [+0.086, +0.125]<br>c [+0.071, +0.146] | +0.061 (+6.1 %)<br>d [+0.020, +0.096]<br>c [-0.006, +0.107] | +0.061 (+6.1 %)<br>d [+0.030, +0.091]<br>c [+0.012, +0.111] | +0.096 (+9.6 %)<br>d [+0.075, +0.115]<br>c [+0.064, +0.122] | +0.099 (+9.9 %)<br>d [+0.082, +0.115]<br>c [+0.073, +0.126] | +0.105 (+10.5 %)<br>d [+0.088, +0.122]<br>c [+0.079, +0.133] |
| block 32002-32310 | 29,156 / 127 / 14 | +0.042 (+4.2 %)<br>d [+0.012, +0.069]<br>c [+0.019, +0.071] | -0.012 (-1.2 %)<br>d [-0.131, +0.055]<br>c [-0.140, +0.064] | +0.008 (+0.8 %)<br>d [-0.048, +0.053]<br>c [-0.058, +0.059] | +0.047 (+4.7 %)<br>d [-0.005, +0.083]<br>c [-0.000, +0.090] | +0.052 (+5.2 %)<br>d [+0.029, +0.076]<br>c [+0.028, +0.081] | +0.136 (+13.6 %)<br>d [+0.107, +0.162]<br>c [+0.083, +0.165] |
| block from 32312 | 18,267 / 83 / 19 | +0.032 (+3.2 %)<br>d [+0.015, +0.051]<br>c [+0.009, +0.054] | -0.016 (-1.6 %)<br>d [-0.046, +0.015]<br>c [-0.072, +0.023] | -0.023 (-2.3 %)<br>d [-0.053, +0.006]<br>c [-0.059, +0.012] | +0.017 (+1.7 %)<br>d [+0.000, +0.035]<br>c [-0.016, +0.041] | +0.018 (+1.8 %)<br>d [+0.000, +0.035]<br>c [-0.005, +0.040] | +0.194 (+19.4 %)<br>d [+0.162, +0.222]<br>c [+0.150, +0.237] |
| BES level fifth 1 | 39,189 / 422 / 78 | +0.038 (+3.8 %)<br>d [+0.010, +0.064]<br>c [+0.004, +0.071] | -0.033 (-3.3 %)<br>d [-0.133, +0.017]<br>c [-0.140, +0.019] | +0.006 (+0.6 %)<br>d [-0.042, +0.042]<br>c [-0.048, +0.057] | +0.017 (+1.7 %)<br>d [-0.029, +0.044]<br>c [-0.030, +0.047] | +0.038 (+3.8 %)<br>d [+0.017, +0.058]<br>c [+0.014, +0.069] | +0.183 (+18.3 %)<br>d [+0.154, +0.210]<br>c [+0.152, +0.213] |
| BES level fifth 2 | 38,652 / 499 / 73 | +0.088 (+8.8 %)<br>d [+0.066, +0.110]<br>c [+0.062, +0.122] | +0.065 (+6.5 %)<br>d [-0.005, +0.123]<br>c [-0.033, +0.123] | +0.043 (+4.3 %)<br>d [-0.008, +0.095]<br>c [-0.018, +0.102] | +0.090 (+9.0 %)<br>d [+0.055, +0.121]<br>c [+0.043, +0.121] | +0.085 (+8.5 %)<br>d [+0.059, +0.115]<br>c [+0.054, +0.120] | +0.120 (+12.0 %)<br>d [+0.094, +0.147]<br>c [+0.087, +0.154] |
| BES level fifth 3 | 42,148 / 422 / 57 | +0.102 (+10.2 %)<br>d [+0.076, +0.129]<br>c [+0.055, +0.161] | +0.079 (+7.9 %)<br>d [+0.048, +0.109]<br>c [+0.030, +0.125] | +0.071 (+7.1 %)<br>d [+0.042, +0.099]<br>c [+0.015, +0.127] | +0.105 (+10.5 %)<br>d [+0.089, +0.122]<br>c [+0.080, +0.131] | +0.104 (+10.4 %)<br>d [+0.088, +0.119]<br>c [+0.075, +0.131] | +0.093 (+9.3 %)<br>d [+0.072, +0.117]<br>c [+0.068, +0.127] |
| BES level fifth 4 | 42,932 / 361 / 54 | +0.093 (+9.3 %)<br>d [+0.071, +0.116]<br>c [+0.062, +0.124] | +0.050 (+5.0 %)<br>d [+0.007, +0.099]<br>c [-0.021, +0.119] | +0.063 (+6.3 %)<br>d [+0.027, +0.097]<br>c [+0.001, +0.119] | +0.107 (+10.7 %)<br>d [+0.081, +0.137]<br>c [+0.064, +0.151] | +0.106 (+10.6 %)<br>d [+0.084, +0.128]<br>c [+0.069, +0.140] | +0.109 (+10.9 %)<br>d [+0.087, +0.132]<br>c [+0.078, +0.140] |
| BES level fifth 5 | 41,812 / 297 / 47 | +0.121 (+12.1 %)<br>d [+0.086, +0.153]<br>c [+0.052, +0.158] | +0.002 (+0.2 %)<br>d [-0.062, +0.053]<br>c [-0.181, +0.072] | +0.012 (+1.2 %)<br>d [-0.027, +0.047]<br>c [-0.082, +0.063] | +0.073 (+7.3 %)<br>d [+0.047, +0.098]<br>c [+0.011, +0.104] | +0.067 (+6.7 %)<br>d [+0.048, +0.088]<br>c [+0.031, +0.102] | +0.055 (+5.5 %)<br>d [+0.015, +0.105]<br>c [+0.008, +0.157] |

### Target L5

| stratum | rows / discharges / clusters | trees TX over T0 | filters KX over K0 | TB over T0 (BES only) | TE over T0 (ECEI only) | TXlag over T0 (one row old) | network over twin, as built | two-stage over two-stage twin, as built | network over twin, gain fitted | two-stage over two-stage twin, gain fitted | T0 over G |
|---|---|---|---|---|---|---|---|---|---|---|---|
| block up to 31991 | 164,126 / 403 / 53 | +0.094 (+9.4 %)<br>d [+0.072, +0.113]<br>c [+0.060, +0.129] | +0.070 (+7.0 %)<br>d [+0.056, +0.084]<br>c [+0.045, +0.101] | +0.082 (+8.2 %)<br>d [+0.065, +0.099]<br>c [+0.054, +0.115] | +0.036 (+3.6 %)<br>d [+0.023, +0.047]<br>c [+0.021, +0.049] | +0.043 (+4.3 %)<br>d [+0.024, +0.060]<br>c [+0.015, +0.068] | +0.039 (+3.9 %)<br>d [-0.004, +0.072]<br>c [-0.030, +0.083] | +0.031 (+3.1 %)<br>d [-0.003, +0.059]<br>c [-0.015, +0.071] | +0.076 (+7.6 %)<br>d [+0.058, +0.093]<br>c [+0.047, +0.096] | +0.074 (+7.4 %)<br>d [+0.060, +0.087]<br>c [+0.055, +0.091] | +0.121 (+12.1 %)<br>d [+0.101, +0.141]<br>c [+0.088, +0.152] |
| block 32002-32310 | 32,087 / 127 / 14 | +0.030 (+3.0 %)<br>d [-0.086, +0.079]<br>c [-0.061, +0.072] | +0.024 (+2.4 %)<br>d [+0.006, +0.040]<br>c [-0.002, +0.050] | +0.025 (+2.5 %)<br>d [-0.074, +0.067]<br>c [-0.060, +0.076] | +0.020 (+2.0 %)<br>d [-0.007, +0.049]<br>c [+0.003, +0.036] | +0.001 (+0.1 %)<br>d [-0.073, +0.039]<br>c [-0.053, +0.034] | -0.036 (-3.6 %)<br>d [-0.127, +0.030]<br>c [-0.133, +0.033] | -0.014 (-1.4 %)<br>d [-0.065, +0.033]<br>c [-0.079, +0.041] | +0.037 (+3.7 %)<br>d [+0.000, +0.069]<br>c [+0.003, +0.069] | +0.043 (+4.3 %)<br>d [+0.019, +0.066]<br>c [+0.021, +0.069] | +0.211 (+21.1 %)<br>d [+0.165, +0.256]<br>c [+0.142, +0.268] |
| block from 32312 | 19,156 / 83 / 19 | +0.047 (+4.7 %)<br>d [+0.026, +0.072]<br>c [+0.023, +0.068] | -0.004 (-0.4 %)<br>d [-0.021, +0.011]<br>c [-0.030, +0.012] | +0.040 (+4.0 %)<br>d [+0.018, +0.059]<br>c [+0.017, +0.061] | +0.035 (+3.5 %)<br>d [+0.016, +0.050]<br>c [+0.015, +0.050] | +0.031 (+3.1 %)<br>d [+0.010, +0.049]<br>c [+0.013, +0.049] | -0.046 (-4.6 %)<br>d [-0.104, +0.005]<br>c [-0.135, +0.017] | -0.065 (-6.5 %)<br>d [-0.115, -0.014]<br>c [-0.120, -0.007] | +0.015 (+1.5 %)<br>d [-0.010, +0.040]<br>c [-0.028, +0.042] | +0.015 (+1.5 %)<br>d [-0.008, +0.040]<br>c [-0.014, +0.045] | +0.335 (+33.5 %)<br>d [+0.276, +0.394]<br>c [+0.264, +0.406] |
| BES level fifth 1 | 41,730 / 446 / 81 | +0.038 (+3.8 %)<br>d [-0.084, +0.087]<br>c [-0.085, +0.094] | +0.023 (+2.3 %)<br>d [+0.005, +0.040]<br>c [-0.000, +0.044] | +0.029 (+2.9 %)<br>d [-0.071, +0.072]<br>c [-0.071, +0.077] | +0.032 (+3.2 %)<br>d [+0.009, +0.053]<br>c [+0.005, +0.059] | +0.014 (+1.4 %)<br>d [-0.075, +0.055]<br>c [-0.076, +0.059] | -0.021 (-2.1 %)<br>d [-0.118, +0.036]<br>c [-0.121, +0.040] | +0.012 (+1.2 %)<br>d [-0.037, +0.055]<br>c [-0.053, +0.071] | +0.033 (+3.3 %)<br>d [-0.009, +0.063]<br>c [-0.008, +0.065] | +0.052 (+5.2 %)<br>d [+0.028, +0.075]<br>c [+0.021, +0.082] | +0.277 (+27.7 %)<br>d [+0.230, +0.323]<br>c [+0.221, +0.328] |
| BES level fifth 2 | 41,900 / 505 / 73 | +0.078 (+7.8 %)<br>d [+0.057, +0.099]<br>c [+0.051, +0.105] | +0.069 (+6.9 %)<br>d [+0.056, +0.082]<br>c [+0.054, +0.086] | +0.068 (+6.8 %)<br>d [+0.050, +0.085]<br>c [+0.048, +0.093] | +0.032 (+3.2 %)<br>d [+0.012, +0.053]<br>c [+0.015, +0.048] | +0.030 (+3.0 %)<br>d [+0.011, +0.049]<br>c [+0.005, +0.053] | +0.042 (+4.2 %)<br>d [-0.029, +0.098]<br>c [-0.052, +0.097] | +0.019 (+1.9 %)<br>d [-0.032, +0.062]<br>c [-0.039, +0.059] | +0.073 (+7.3 %)<br>d [+0.043, +0.103]<br>c [+0.033, +0.100] | +0.066 (+6.6 %)<br>d [+0.045, +0.089]<br>c [+0.040, +0.087] | +0.144 (+14.4 %)<br>d [+0.112, +0.176]<br>c [+0.103, +0.184] |
| BES level fifth 3 | 43,717 / 431 / 58 | +0.102 (+10.2 %)<br>d [+0.074, +0.130]<br>c [+0.050, +0.157] | +0.079 (+7.9 %)<br>d [+0.059, +0.100]<br>c [+0.041, +0.124] | +0.087 (+8.7 %)<br>d [+0.060, +0.115]<br>c [+0.040, +0.142] | +0.043 (+4.3 %)<br>d [+0.029, +0.057]<br>c [+0.024, +0.061] | +0.052 (+5.2 %)<br>d [+0.028, +0.074]<br>c [+0.016, +0.084] | +0.014 (+1.4 %)<br>d [-0.029, +0.051]<br>c [-0.058, +0.075] | +0.015 (+1.5 %)<br>d [-0.024, +0.050]<br>c [-0.057, +0.085] | +0.064 (+6.4 %)<br>d [+0.045, +0.082]<br>c [+0.033, +0.091] | +0.067 (+6.7 %)<br>d [+0.051, +0.083]<br>c [+0.038, +0.096] | +0.114 (+11.4 %)<br>d [+0.086, +0.143]<br>c [+0.074, +0.156] |
| BES level fifth 4 | 44,117 / 366 / 54 | +0.072 (+7.2 %)<br>d [+0.049, +0.094]<br>c [+0.041, +0.101] | +0.062 (+6.2 %)<br>d [+0.037, +0.085]<br>c [+0.008, +0.108] | +0.063 (+6.3 %)<br>d [+0.042, +0.084]<br>c [+0.032, +0.092] | +0.019 (+1.9 %)<br>d [+0.002, +0.036]<br>c [+0.004, +0.035] | +0.016 (+1.6 %)<br>d [-0.008, +0.034]<br>c [-0.018, +0.041] | +0.006 (+0.6 %)<br>d [-0.044, +0.055]<br>c [-0.080, +0.078] | +0.013 (+1.3 %)<br>d [-0.027, +0.049]<br>c [-0.061, +0.072] | +0.069 (+6.9 %)<br>d [+0.045, +0.098]<br>c [+0.028, +0.110] | +0.066 (+6.6 %)<br>d [+0.048, +0.084]<br>c [+0.034, +0.094] | +0.111 (+11.1 %)<br>d [+0.079, +0.143]<br>c [+0.060, +0.163] |
| BES level fifth 5 | 43,905 / 305 / 49 | +0.105 (+10.5 %)<br>d [+0.066, +0.135]<br>c [+0.035, +0.142] | +0.005 (+0.5 %)<br>d [-0.013, +0.022]<br>c [-0.053, +0.044] | +0.100 (+10.0 %)<br>d [+0.071, +0.126]<br>c [+0.040, +0.131] | +0.028 (+2.8 %)<br>d [+0.009, +0.051]<br>c [+0.002, +0.066] | +0.071 (+7.1 %)<br>d [+0.037, +0.098]<br>c [-0.010, +0.110] | +0.012 (+1.2 %)<br>d [-0.066, +0.073]<br>c [-0.224, +0.104] | -0.012 (-1.2 %)<br>d [-0.061, +0.030]<br>c [-0.136, +0.047] | +0.075 (+7.5 %)<br>d [+0.044, +0.102]<br>c [-0.014, +0.111] | +0.052 (+5.2 %)<br>d [+0.032, +0.072]<br>c [+0.004, +0.078] | +0.113 (+11.3 %)<br>d [+0.075, +0.162]<br>c [+0.065, +0.201] |

### Target L10

| stratum | rows / discharges / clusters | trees TX over T0 | network over twin, as built | two-stage over two-stage twin, as built | network over twin, gain fitted | two-stage over two-stage twin, gain fitted | T0 over G |
|---|---|---|---|---|---|---|---|
| block up to 31991 | 164,177 / 402 / 53 | +0.086 (+8.6 %)<br>d [+0.064, +0.107]<br>c [+0.057, +0.115] | +0.022 (+2.2 %)<br>d [-0.014, +0.052]<br>c [-0.035, +0.071] | -0.003 (-0.3 %)<br>d [-0.032, +0.020]<br>c [-0.039, +0.034] | +0.064 (+6.4 %)<br>d [+0.050, +0.078]<br>c [+0.041, +0.086] | +0.053 (+5.3 %)<br>d [+0.044, +0.062]<br>c [+0.041, +0.065] | +0.159 (+15.9 %)<br>d [+0.136, +0.183]<br>c [+0.120, +0.194] |
| block 32002-32310 | 32,400 / 127 / 14 | +0.030 (+3.0 %)<br>d [-0.107, +0.094]<br>c [-0.074, +0.082] | -0.036 (-3.6 %)<br>d [-0.130, +0.036]<br>c [-0.134, +0.042] | -0.023 (-2.3 %)<br>d [-0.082, +0.026]<br>c [-0.099, +0.044] | +0.042 (+4.2 %)<br>d [+0.005, +0.079]<br>c [+0.008, +0.078] | +0.042 (+4.2 %)<br>d [+0.019, +0.070]<br>c [+0.020, +0.077] | +0.282 (+28.2 %)<br>d [+0.226, +0.342]<br>c [+0.180, +0.365] |
| block from 32312 | 19,090 / 82 / 19 | +0.073 (+7.3 %)<br>d [+0.038, +0.108]<br>c [+0.034, +0.118] | -0.072 (-7.2 %)<br>d [-0.155, +0.005]<br>c [-0.181, +0.028] | -0.111 (-11.1 %)<br>d [-0.175, -0.057]<br>c [-0.174, -0.049] | +0.012 (+1.2 %)<br>d [-0.023, +0.047]<br>c [-0.037, +0.053] | +0.008 (+0.8 %)<br>d [-0.017, +0.031]<br>c [-0.021, +0.034] | +0.418 (+41.8 %)<br>d [+0.355, +0.478]<br>c [+0.335, +0.491] |
| BES level fifth 1 | 41,774 / 440 / 81 | +0.045 (+4.5 %)<br>d [-0.115, +0.115]<br>c [-0.118, +0.118] | -0.026 (-2.6 %)<br>d [-0.121, +0.039]<br>c [-0.130, +0.041] | +0.001 (+0.1 %)<br>d [-0.054, +0.049]<br>c [-0.077, +0.063] | +0.037 (+3.7 %)<br>d [-0.002, +0.070]<br>c [-0.002, +0.069] | +0.050 (+5.0 %)<br>d [+0.026, +0.081]<br>c [+0.019, +0.084] | +0.338 (+33.8 %)<br>d [+0.285, +0.391]<br>c [+0.261, +0.412] |
| BES level fifth 2 | 42,225 / 502 / 75 | +0.081 (+8.1 %)<br>d [+0.058, +0.104]<br>c [+0.048, +0.111] | +0.023 (+2.3 %)<br>d [-0.041, +0.076]<br>c [-0.061, +0.085] | -0.016 (-1.6 %)<br>d [-0.064, +0.022]<br>c [-0.069, +0.026] | +0.061 (+6.1 %)<br>d [+0.036, +0.087]<br>c [+0.027, +0.089] | +0.047 (+4.7 %)<br>d [+0.030, +0.063]<br>c [+0.027, +0.063] | +0.184 (+18.4 %)<br>d [+0.144, +0.223]<br>c [+0.125, +0.240] |
| BES level fifth 3 | 43,693 / 427 / 58 | +0.092 (+9.2 %)<br>d [+0.066, +0.119]<br>c [+0.045, +0.136] | -0.003 (-0.3 %)<br>d [-0.045, +0.035]<br>c [-0.072, +0.054] | -0.014 (-1.4 %)<br>d [-0.049, +0.018]<br>c [-0.076, +0.045] | +0.053 (+5.3 %)<br>d [+0.035, +0.071]<br>c [+0.025, +0.078] | +0.051 (+5.1 %)<br>d [+0.038, +0.064]<br>c [+0.030, +0.071] | +0.169 (+16.9 %)<br>d [+0.140, +0.199]<br>c [+0.115, +0.221] |
| BES level fifth 4 | 44,019 / 365 / 53 | +0.049 (+4.9 %)<br>d [+0.031, +0.067]<br>c [+0.026, +0.069] | -0.023 (-2.3 %)<br>d [-0.064, +0.016]<br>c [-0.099, +0.032] | -0.026 (-2.6 %)<br>d [-0.061, +0.005]<br>c [-0.094, +0.026] | +0.048 (+4.8 %)<br>d [+0.029, +0.069]<br>c [+0.017, +0.074] | +0.040 (+4.0 %)<br>d [+0.026, +0.054]<br>c [+0.017, +0.060] | +0.147 (+14.7 %)<br>d [+0.113, +0.180]<br>c [+0.093, +0.199] |
| BES level fifth 5 | 43,956 / 306 / 48 | +0.105 (+10.5 %)<br>d [+0.068, +0.135]<br>c [+0.030, +0.153] | +0.039 (+3.9 %)<br>d [-0.054, +0.113]<br>c [-0.216, +0.166] | -0.030 (-3.0 %)<br>d [-0.087, +0.020]<br>c [-0.160, +0.040] | +0.090 (+9.0 %)<br>d [+0.052, +0.124]<br>c [-0.011, +0.146] | +0.047 (+4.7 %)<br>d [+0.025, +0.067]<br>c [-0.000, +0.076] | +0.159 (+15.9 %)<br>d [+0.114, +0.207]<br>c [+0.102, +0.238] |

## Addendum 1 (session lead): strata on the inputs, the rotation, rows without beam emission

Implemented as in Addendum 2 section 12. The thresholds were set by a reviewer after looking at the data and are taken over unchanged. Each estimator keeps its fit on all fit rows of its target and is only scored by class. Share of the F-reduction = (SSE(B) - SSE(A)) on the class's F rows over the same on all F rows.

### A1: by the BES sample relative to its recent level (r_B), V1 random (r2-r4)

| class | share of rows (F rows) | rows F / L5 | trees TX over T0: against F | against L5 | share of the F-reduction | filters KX over K0: against F | against L5 | share of the F-reduction | two-stage over two-stage twin, as built: against F | against L5 | share of the F-reduction | network over twin, as built: against F | against L5 | share of the F-reduction | two-stage over two-stage twin, gain fitted: against F | against L5 | share of the F-reduction | network over twin, gain fitted: against F | against L5 | share of the F-reduction |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| below 0.9 | 0.048 | 10,798 / 9,486 | +0.129 (+12.9 %)<br>d [+0.098, +0.161]<br>c [+0.080, +0.188] | +0.183 (+18.3 %)<br>d [+0.139, +0.227]<br>c [+0.124, +0.254] | +0.251<br>d [+0.210, +0.293]<br>c [+0.186, +0.326] | +0.078 (+7.8 %)<br>d [+0.058, +0.101]<br>c [+0.044, +0.124] | +0.165 (+16.5 %)<br>d [+0.132, +0.194]<br>c [+0.120, +0.211] | +0.270<br>d [+0.227, +0.313]<br>c [+0.200, +0.348] | +0.136 (+13.6 %)<br>d [+0.100, +0.172]<br>c [+0.082, +0.197] | +0.147 (+14.7 %)<br>d [+0.101, +0.190]<br>c [+0.085, +0.201] | +0.238<br>d [+0.188, +0.292]<br>c [+0.171, +0.320] | +0.086 (+8.6 %)<br>d [+0.051, +0.121]<br>c [+0.045, +0.137] | +0.143 (+14.3 %)<br>d [+0.080, +0.195]<br>c [+0.074, +0.201] | +0.277<br>d [+0.174, +0.503]<br>c [+0.157, +0.630] | +0.136 (+13.6 %)<br>d [+0.104, +0.169]<br>c [+0.087, +0.193] | +0.147 (+14.7 %)<br>d [+0.121, +0.180]<br>c [+0.114, +0.182] | +0.233<br>d [+0.190, +0.279]<br>c [+0.174, +0.304] | +0.083 (+8.3 %)<br>d [+0.057, +0.110]<br>c [+0.050, +0.123] | +0.124 (+12.4 %)<br>d [+0.087, +0.161]<br>c [+0.083, +0.162] | +0.214<br>d [+0.160, +0.282]<br>c [+0.148, +0.302] |
| 0.9-0.97 | 0.176 | 39,160 / 38,056 | +0.090 (+9.0 %)<br>d [+0.066, +0.115]<br>c [+0.051, +0.137] | +0.071 (+7.1 %)<br>d [+0.044, +0.094]<br>c [+0.046, +0.097] | +0.152<br>d [+0.120, +0.181]<br>c [+0.103, +0.192] | +0.053 (+5.3 %)<br>d [+0.042, +0.064]<br>c [+0.032, +0.074] | +0.075 (+7.5 %)<br>d [+0.062, +0.087]<br>c [+0.053, +0.096] | +0.156<br>d [+0.129, +0.183]<br>c [+0.108, +0.199] | +0.100 (+10.0 %)<br>d [+0.072, +0.129]<br>c [+0.052, +0.152] | +0.049 (+4.9 %)<br>d [+0.016, +0.080]<br>c [+0.003, +0.093] | +0.152<br>d [+0.116, +0.187]<br>c [+0.094, +0.200] | +0.040 (+4.0 %)<br>d [+0.006, +0.074]<br>c [-0.011, +0.084] | +0.057 (+5.7 %)<br>d [+0.022, +0.088]<br>c [+0.009, +0.093] | +0.108<br>d [+0.025, +0.179]<br>c [-0.053, +0.190] | +0.102 (+10.2 %)<br>d [+0.077, +0.129]<br>c [+0.060, +0.149] | +0.079 (+7.9 %)<br>d [+0.066, +0.093]<br>c [+0.061, +0.098] | +0.152<br>d [+0.121, +0.182]<br>c [+0.102, +0.195] | +0.054 (+5.4 %)<br>d [+0.033, +0.078]<br>c [+0.023, +0.086] | +0.078 (+7.8 %)<br>d [+0.061, +0.094]<br>c [+0.056, +0.097] | +0.122<br>d [+0.084, +0.164]<br>c [+0.065, +0.172] |
| 0.97-1.03 | 0.433 | 96,514 / 94,686 | +0.055 (+5.5 %)<br>d [+0.042, +0.068]<br>c [+0.032, +0.080] | +0.036 (+3.6 %)<br>d [-0.000, +0.055]<br>c [-0.002, +0.061] | +0.151<br>d [+0.123, +0.181]<br>c [+0.105, +0.209] | +0.011 (+1.1 %)<br>d [+0.007, +0.015]<br>c [+0.004, +0.018] | +0.009 (+0.9 %)<br>d [+0.005, +0.013]<br>c [+0.002, +0.016] | +0.051<br>d [+0.034, +0.069]<br>c [+0.023, +0.083] | +0.056 (+5.6 %)<br>d [+0.035, +0.075]<br>c [+0.024, +0.087] | +0.003 (+0.3 %)<br>d [-0.029, +0.028]<br>c [-0.035, +0.032] | +0.139<br>d [+0.095, +0.177]<br>c [+0.073, +0.204] | +0.013 (+1.3 %)<br>d [-0.022, +0.038]<br>c [-0.036, +0.044] | -0.019 (-1.9 %)<br>d [-0.075, +0.019]<br>c [-0.097, +0.032] | +0.054<br>d [-0.163, +0.145]<br>c [-0.339, +0.172] | +0.058 (+5.8 %)<br>d [+0.041, +0.074]<br>c [+0.031, +0.085] | +0.037 (+3.7 %)<br>d [+0.024, +0.049]<br>c [+0.022, +0.050] | +0.141<br>d [+0.105, +0.174]<br>c [+0.087, +0.199] | +0.032 (+3.2 %)<br>d [+0.013, +0.047]<br>c [+0.005, +0.051] | +0.035 (+3.5 %)<br>d [+0.014, +0.052]<br>c [+0.004, +0.058] | +0.116<br>d [+0.055, +0.160]<br>c [+0.027, +0.184] |
| above 1.03 | 0.256 | 57,085 / 54,642 | +0.099 (+9.9 %)<br>d [+0.082, +0.116]<br>c [+0.068, +0.133] | +0.065 (+6.5 %)<br>d [+0.034, +0.089]<br>c [+0.027, +0.098] | +0.372<br>d [+0.332, +0.416]<br>c [+0.303, +0.458] | +0.055 (+5.5 %)<br>d [+0.043, +0.066]<br>c [+0.032, +0.077] | +0.017 (+1.7 %)<br>d [+0.001, +0.032]<br>c [-0.008, +0.046] | +0.370<br>d [+0.324, +0.414]<br>c [+0.281, +0.447] | +0.112 (+11.2 %)<br>d [+0.085, +0.138]<br>c [+0.070, +0.157] | -0.064 (-6.4 %)<br>d [-0.116, -0.022]<br>c [-0.133, -0.006] | +0.386<br>d [+0.326, +0.446]<br>c [+0.296, +0.490] | +0.072 (+7.2 %)<br>d [+0.038, +0.103]<br>c [+0.028, +0.110] | -0.034 (-3.4 %)<br>d [-0.094, +0.012]<br>c [-0.105, +0.026] | +0.469<br>d [+0.354, +0.610]<br>c [+0.322, +0.712] | +0.113 (+11.3 %)<br>d [+0.090, +0.136]<br>c [+0.076, +0.154] | +0.027 (+2.7 %)<br>d [+0.008, +0.046]<br>c [-0.004, +0.054] | +0.381<br>d [+0.329, +0.433]<br>c [+0.302, +0.471] | +0.080 (+8.0 %)<br>d [+0.059, +0.103]<br>c [+0.050, +0.109] | +0.041 (+4.1 %)<br>d [+0.017, +0.064]<br>c [+0.008, +0.071] | +0.425<br>d [+0.361, +0.493]<br>c [+0.341, +0.517] |
| r_B undefined | 0.086 | 19,115 / 18,499 | +0.055 (+5.5 %)<br>d [+0.031, +0.079]<br>c [+0.026, +0.090] | +0.063 (+6.3 %)<br>d [-0.031, +0.105]<br>c [-0.034, +0.111] | +0.073<br>d [+0.042, +0.103]<br>c [+0.043, +0.106] | +0.064 (+6.4 %)<br>d [+0.045, +0.084]<br>c [+0.033, +0.099] | +0.092 (+9.2 %)<br>d [+0.071, +0.114]<br>c [+0.058, +0.125] | +0.153<br>d [+0.116, +0.190]<br>c [+0.104, +0.210] | +0.075 (+7.5 %)<br>d [+0.038, +0.109]<br>c [+0.029, +0.120] | +0.025 (+2.5 %)<br>d [-0.029, +0.074]<br>c [-0.044, +0.081] | +0.084<br>d [+0.043, +0.124]<br>c [+0.038, +0.124] | +0.042 (+4.2 %)<br>d [-0.049, +0.093]<br>c [-0.048, +0.108] | +0.021 (+2.1 %)<br>d [-0.134, +0.093]<br>c [-0.151, +0.111] | +0.092<br>d [-0.098, +0.201]<br>c [-0.119, +0.224] | +0.084 (+8.4 %)<br>d [+0.051, +0.114]<br>c [+0.044, +0.124] | +0.076 (+7.6 %)<br>d [+0.052, +0.098]<br>c [+0.049, +0.099] | +0.093<br>d [+0.057, +0.127]<br>c [+0.055, +0.127] | +0.070 (+7.0 %)<br>d [+0.013, +0.104]<br>c [+0.014, +0.117] | +0.084 (+8.4 %)<br>d [+0.020, +0.115]<br>c [+0.017, +0.121] | +0.122<br>d [+0.026, +0.177]<br>c [+0.024, +0.185] |

### A1: by the BES sample relative to its recent level (r_B), V4 chronological: fit to 31991, score from 32312

| class | share of rows (F rows) | rows F / L5 | trees TX over T0: against F | against L5 | share of the F-reduction |
|---|---|---|---|---|---|
| below 0.9 | 0.033 | 665 / 564 | +0.024 (+2.4 %)<br>d [-0.020, +0.074]<br>c [-0.059, +0.108] | +0.247 (+24.7 %)<br>d [+0.168, +0.334]<br>c [+0.178, +0.318] | +0.092<br>d [-0.154, +0.445]<br>c [-0.615, +0.290] |
| 0.9-0.97 | 0.173 | 3,470 / 3,368 | +0.017 (+1.7 %)<br>d [-0.021, +0.057]<br>c [-0.029, +0.067] | +0.101 (+10.1 %)<br>d [+0.034, +0.169]<br>c [+0.030, +0.185] | +0.091<br>d [-0.291, +0.370]<br>c [-0.436, +0.392] |
| 0.97-1.03 | 0.459 | 9,207 / 8,937 | +0.005 (+0.5 %)<br>d [-0.018, +0.029]<br>c [-0.019, +0.031] | +0.122 (+12.2 %)<br>d [+0.070, +0.175]<br>c [+0.046, +0.200] | +0.079<br>d [-0.687, +0.548]<br>c [-0.582, +0.774] |
| above 1.03 | 0.257 | 5,151 / 4,885 | +0.031 (+3.1 %)<br>d [+0.003, +0.063]<br>c [+0.004, +0.056] | +0.047 (+4.7 %)<br>d [+0.001, +0.103]<br>c [+0.016, +0.078] | +0.672<br>d [+0.113, +1.471]<br>c [+0.162, +1.829] |
| r_B undefined | 0.078 | 1,569 / 1,402 | +0.010 (+1.0 %)<br>d [-0.005, +0.027]<br>c [-0.005, +0.019] | +0.022 (+2.2 %)<br>d [-0.003, +0.049]<br>c [+0.004, +0.036] | +0.067<br>d [-0.068, +0.274]<br>c [-0.048, +0.288] |

### A1: by the BES sample relative to its recent level (r_B), V4 chronological: fit to 32310, score from 32312

| class | share of rows (F rows) | rows F / L5 | trees TX over T0: against F | against L5 | share of the F-reduction |
|---|---|---|---|---|---|
| below 0.9 | 0.033 | 665 / 564 | -0.004 (-0.4 %)<br>d [-0.063, +0.060]<br>c [-0.084, +0.110] | +0.215 (+21.5 %)<br>d [+0.111, +0.336]<br>c [+0.091, +0.318] | -0.009<br>d [-0.249, +0.138]<br>c [-0.387, +0.252] |
| 0.9-0.97 | 0.173 | 3,470 / 3,368 | +0.015 (+1.5 %)<br>d [-0.022, +0.054]<br>c [-0.017, +0.051] | +0.084 (+8.4 %)<br>d [+0.033, +0.141]<br>c [+0.030, +0.173] | +0.050<br>d [-0.151, +0.144]<br>c [-0.117, +0.140] |
| 0.97-1.03 | 0.459 | 9,207 / 8,937 | +0.016 (+1.6 %)<br>d [+0.002, +0.031]<br>c [+0.004, +0.034] | +0.081 (+8.1 %)<br>d [+0.049, +0.115]<br>c [+0.042, +0.122] | +0.152<br>d [+0.028, +0.316]<br>c [+0.043, +0.509] |
| above 1.03 | 0.257 | 5,151 / 4,885 | +0.061 (+6.1 %)<br>d [+0.028, +0.088]<br>c [+0.016, +0.084] | +0.062 (+6.2 %)<br>d [+0.024, +0.098]<br>c [+0.024, +0.095] | +0.786<br>d [+0.566, +1.195]<br>c [+0.423, +1.116] |
| r_B undefined | 0.078 | 1,569 / 1,402 | +0.005 (+0.5 %)<br>d [-0.021, +0.027]<br>c [-0.026, +0.021] | +0.007 (+0.7 %)<br>d [-0.007, +0.020]<br>c [-0.012, +0.016] | +0.021<br>d [-0.106, +0.123]<br>c [-0.080, +0.170] |

### A2: by the position of the row relative to absent frames, V1 random (r2-r4)

| class | share of rows (F rows) | rows F / L5 | trees TX over T0: against F | against L5 | share of the F-reduction | filters KX over K0: against F | against L5 | share of the F-reduction | two-stage over two-stage twin, as built: against F | against L5 | share of the F-reduction | network over twin, as built: against F | against L5 | share of the F-reduction | two-stage over two-stage twin, gain fitted: against F | against L5 | share of the F-reduction | network over twin, gain fitted: against F | against L5 | share of the F-reduction |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| absent frames earlier and later | 0.467 | 103,937 / 102,578 | +0.130 (+13.0 %)<br>d [+0.104, +0.156]<br>c [+0.087, +0.184] | +0.082 (+8.2 %)<br>d [+0.026, +0.111]<br>c [+0.027, +0.121] | +0.484<br>d [+0.411, +0.556]<br>c [+0.351, +0.582] | +0.084 (+8.4 %)<br>d [+0.070, +0.099]<br>c [+0.057, +0.110] | +0.058 (+5.8 %)<br>d [+0.044, +0.074]<br>c [+0.031, +0.093] | +0.554<br>d [+0.481, +0.628]<br>c [+0.415, +0.672] | +0.135 (+13.5 %)<br>d [+0.100, +0.168]<br>c [+0.078, +0.195] | +0.007 (+0.7 %)<br>d [-0.024, +0.034]<br>c [-0.037, +0.045] | +0.456<br>d [+0.367, +0.543]<br>c [+0.305, +0.564] | +0.070 (+7.0 %)<br>d [+0.032, +0.106]<br>c [+0.013, +0.120] | +0.005 (+0.5 %)<br>d [-0.038, +0.040]<br>c [-0.062, +0.056] | +0.437<br>d [+0.260, +0.638]<br>c [+0.147, +0.653] | +0.136 (+13.6 %)<br>d [+0.106, +0.165]<br>c [+0.087, +0.188] | +0.063 (+6.3 %)<br>d [+0.049, +0.076]<br>c [+0.045, +0.080] | +0.448<br>d [+0.370, +0.524]<br>c [+0.315, +0.547] | +0.084 (+8.4 %)<br>d [+0.060, +0.109]<br>c [+0.047, +0.118] | +0.062 (+6.2 %)<br>d [+0.043, +0.080]<br>c [+0.034, +0.084] | +0.429<br>d [+0.336, +0.522]<br>c [+0.287, +0.545] |
| only later (before the first) | 0.261 | 58,120 / 56,222 | +0.072 (+7.2 %)<br>d [+0.056, +0.091]<br>c [+0.048, +0.102] | +0.061 (+6.1 %)<br>d [+0.046, +0.076]<br>c [+0.039, +0.082] | +0.224<br>d [+0.174, +0.281]<br>c [+0.149, +0.329] | +0.038 (+3.8 %)<br>d [+0.025, +0.052]<br>c [+0.016, +0.063] | +0.059 (+5.9 %)<br>d [+0.048, +0.069]<br>c [+0.043, +0.074] | +0.204<br>d [+0.144, +0.264]<br>c [+0.107, +0.321] | +0.090 (+9.0 %)<br>d [+0.067, +0.116]<br>c [+0.055, +0.132] | +0.031 (+3.1 %)<br>d [+0.000, +0.058]<br>c [-0.018, +0.075] | +0.246<br>d [+0.186, +0.322]<br>c [+0.153, +0.387] | +0.060 (+6.0 %)<br>d [+0.036, +0.086]<br>c [+0.029, +0.094] | +0.042 (+4.2 %)<br>d [-0.003, +0.086]<br>c [-0.013, +0.093] | +0.302<br>d [+0.193, +0.523]<br>c [+0.157, +0.734] | +0.093 (+9.3 %)<br>d [+0.071, +0.116]<br>c [+0.060, +0.132] | +0.068 (+6.8 %)<br>d [+0.053, +0.081]<br>c [+0.046, +0.088] | +0.247<br>d [+0.193, +0.311]<br>c [+0.161, +0.366] | +0.067 (+6.7 %)<br>d [+0.049, +0.086]<br>c [+0.044, +0.094] | +0.071 (+7.1 %)<br>d [+0.048, +0.094]<br>c [+0.044, +0.097] | +0.272<br>d [+0.211, +0.353]<br>c [+0.184, +0.407] |
| only earlier (after the last) | 0.224 | 49,943 / 46,763 | +0.073 (+7.3 %)<br>d [+0.054, +0.092]<br>c [+0.046, +0.105] | +0.088 (+8.8 %)<br>d [+0.057, +0.116]<br>c [+0.052, +0.127] | +0.278<br>d [+0.222, +0.333]<br>c [+0.226, +0.338] | +0.035 (+3.5 %)<br>d [+0.025, +0.046]<br>c [+0.020, +0.054] | +0.051 (+5.1 %)<br>d [+0.029, +0.073]<br>c [+0.021, +0.090] | +0.237<br>d [+0.187, +0.286]<br>c [+0.188, +0.285] | +0.082 (+8.2 %)<br>d [+0.049, +0.110]<br>c [+0.044, +0.123] | +0.004 (+0.4 %)<br>d [-0.054, +0.045]<br>c [-0.059, +0.053] | +0.284<br>d [+0.187, +0.366]<br>c [+0.200, +0.365] | +0.047 (+4.7 %)<br>d [-0.002, +0.079]<br>c [-0.009, +0.085] | +0.012 (+1.2 %)<br>d [-0.062, +0.060]<br>c [-0.076, +0.072] | +0.298<br>d [-0.037, +0.461]<br>c [-0.105, +0.514] | +0.085 (+8.5 %)<br>d [+0.057, +0.110]<br>c [+0.051, +0.122] | +0.055 (+5.5 %)<br>d [+0.032, +0.076]<br>c [+0.028, +0.080] | +0.285<br>d [+0.207, +0.356]<br>c [+0.218, +0.356] | +0.057 (+5.7 %)<br>d [+0.029, +0.080]<br>c [+0.024, +0.086] | +0.058 (+5.8 %)<br>d [+0.030, +0.084]<br>c [+0.022, +0.091] | +0.296<br>d [+0.178, +0.384]<br>c [+0.176, +0.396] |
| file without absent frames | 0.048 | 10,672 / 9,806 | +0.016 (+1.6 %)<br>d [-0.014, +0.048]<br>c [-0.021, +0.055] | +0.032 (+3.2 %)<br>d [-0.159, +0.092]<br>c [-0.155, +0.096] | +0.014<br>d [-0.013, +0.043]<br>c [-0.020, +0.057] | +0.004 (+0.4 %)<br>d [-0.013, +0.020]<br>c [-0.015, +0.022] | +0.023 (+2.3 %)<br>d [-0.026, +0.059]<br>c [-0.024, +0.060] | +0.006<br>d [-0.019, +0.028]<br>c [-0.023, +0.035] | +0.017 (+1.7 %)<br>d [-0.028, +0.055]<br>c [-0.025, +0.060] | +0.016 (+1.6 %)<br>d [-0.103, +0.095]<br>c [-0.100, +0.104] | +0.013<br>d [-0.020, +0.043]<br>c [-0.019, +0.049] | -0.028 (-2.8 %)<br>d [-0.171, +0.027]<br>c [-0.168, +0.029] | -0.018 (-1.8 %)<br>d [-0.361, +0.133]<br>c [-0.336, +0.137] | -0.037<br>d [-0.243, +0.038]<br>c [-0.242, +0.037] | +0.026 (+2.6 %)<br>d [-0.010, +0.059]<br>c [-0.007, +0.064] | +0.062 (+6.2 %)<br>d [+0.017, +0.102]<br>c [+0.019, +0.106] | +0.020<br>d [-0.006, +0.046]<br>c [-0.005, +0.052] | +0.002 (+0.2 %)<br>d [-0.088, +0.038]<br>c [-0.084, +0.040] | +0.046 (+4.6 %)<br>d [-0.092, +0.117]<br>c [-0.077, +0.117] | +0.003<br>d [-0.097, +0.045]<br>c [-0.088, +0.043] |

### A3: T_i, fresh V_rot and the ECEI level against the BES sample (no estimator)

Regular rows with r_B defined: 223,633 (r_B below 0.40: 362, not in the table). Delivered rotation values equal to the previous row's: 0.550.

| r_B class | rows with T_i ratio | median T_i ratio | rows with fresh V_rot ratio | median V_rot ratio | median ECEI ratio |
|---|---|---|---|---|---|
| 0.40-0.60 | 296 | 0.500 | 209 | 1.006 | 0.867 |
| 0.60-0.80 | 2,113 | 0.663 | 1,326 | 1.006 | 0.912 |
| 0.80-0.90 | 7,914 | 0.843 | 3,293 | 1.003 | 0.953 |
| 0.90-0.97 | 38,888 | 0.962 | 13,877 | 1.000 | 0.977 |
| 0.97-1.03 | 95,978 | 1.003 | 33,750 | 1.000 | 1.000 |
| 1.03-1.10 | 39,586 | 1.053 | 13,792 | 1.002 | 1.026 |
| 1.10-1.20 | 9,805 | 1.159 | 3,769 | 1.005 | 1.047 |
| 1.20-1.40 | 3,518 | 1.334 | 1,620 | 1.008 | 1.053 |
| above 1.40 | 1,317 | 1.582 | 682 | 1.002 | 1.044 |

| slope of the log ratio on log r_B | rows / discharges | T_i | V_rot |
|---|---|---|---|
| C18 form: r_B 0.6-1.6, log ratios clipped at 1.5 | 64,822 / 371 | 1.285 [1.147, 1.422] | +0.024 [-0.009, +0.058] |
| all rows with both positive ratios, no clip | 65,077 / 372 | 0.967 [0.798, 1.155] | -0.004 [-0.032, +0.022] |

### A4: rows exposed without beam emission

Rows whose nine BES channels are all below 0.4 of their median over the ten rows before: 325 (of 239,273 rows examined) in 124 discharges; 210 carry a T_i value (cut, held-free grid); 206 are D1 rows.

| target | rows scored (removed) | TX over T0, refitted without those rows | TX over T0, fits of the main table, same rows |
|---|---|---|---|
| F | 222,466 (206) | +0.088 (+8.8 %)<br>d [+0.074, +0.103]<br>c [+0.061, +0.123] | +0.087 (+8.7 %)<br>d [+0.073, +0.102]<br>c [+0.059, +0.122] |
| L5 | 215,202 (167) | +0.078 (+7.8 %)<br>d [+0.049, +0.097]<br>c [+0.043, +0.108] | +0.075 (+7.5 %)<br>d [+0.044, +0.095]<br>c [+0.039, +0.106] |

## Fitted gains

λ (V1; least-squares slope through the origin of (target - history-only estimate) on the correction, fitted on the other four folds): mean [smallest, largest] over the 15 folds.

| pair | F | N1 | L3 | L5 | L10 |
|---|---|---|---|---|---|
| network over twin (c - t) | 0.695 [0.632, 0.764] | 0.578 [0.531, 0.628] | 0.578 [0.534, 0.624] | 0.536 [0.472, 0.592] | 0.510 [0.448, 0.568] |
| two-stage over two-stage twin (c2 - t2) | 0.864 [0.837, 0.916] | 0.607 [0.588, 0.645] | 0.584 [0.565, 0.616] | 0.529 [0.485, 0.568] | 0.467 [0.418, 0.504] |

Filter gains of KX (b_c: common-mode BES drive; b_g: radial-difference drive), mean [smallest, largest] over the units of a design:

| design | units | b_c fitted to F | b_g fitted to F | b_c fitted to L5 | b_g fitted to L5 |
|---|---|---|---|---|---|
| V1 | 15 | 0.501 [0.473, 0.556] | 0.454 [0.434, 0.478] | 0.525 [0.471, 0.590] | 0.182 [0.147, 0.205] |
| V2 | 15 | 0.502 [0.383, 0.632] | 0.452 [0.375, 0.532] | 0.522 [0.464, 0.589] | 0.182 [0.139, 0.229] |
| V3 | 4 | 0.447 [0.246, 0.637] | 0.390 [0.175, 0.537] | 0.582 [0.399, 0.885] | 0.152 [-0.037, 0.240] |
| V4 | 2 | 0.505 [0.491, 0.520] | 0.421 [0.402, 0.439] | 0.540 [0.529, 0.550] | 0.192 [0.187, 0.196] |

## Usability of the frame (classification)

Label: grid row without a usable T_i, among grid rows of D1 discharges with at least one measured value in the ten rows before. Positives 16,010: missing in the file 14,894, above 3 keV 1,115, held repeat 1. Rows without the fast scalars are left out of both classifiers (example units: {'V1': 5564, 'V4a': 4492}).

| design | rows | positives | AUC history | AUC history + fast | difference |
|---|---|---|---|---|---|
| V1 random (r2-r4) | 235,228 | 15,712 | 0.9285<br>d [0.9208, 0.9356]<br>c [0.9121, 0.9423] | 0.9622<br>d [0.9577, 0.9664]<br>c [0.9506, 0.9719] | +0.0337<br>d [+0.0291, +0.0388]<br>c [+0.0257, +0.0431] |
| V4 chronological: fit to 31991, score from 32312 | 20,914 | 1,601 | 0.9242<br>d [0.9050, 0.9393]<br>c [0.9023, 0.9445] | 0.9477<br>d [0.9358, 0.9573]<br>c [0.9299, 0.9602] | +0.0235<br>d [+0.0131, +0.0361]<br>c [+0.0120, +0.0365] |
| V4 chronological: fit to 32310, score from 32312 | 20,914 | 1,601 | 0.9224<br>d [0.9017, 0.9386]<br>c [0.8989, 0.9442] | 0.9496<br>d [0.9382, 0.9588]<br>c [0.9313, 0.9613] | +0.0272<br>d [+0.0158, +0.0422]<br>c [+0.0128, +0.0461] |

## V3: each scored fifth (points)

| unit (fifth scored) | target | rows | TX over T0 | KX over K0 | T0 over G |
|---|---|---|---|---|---|
| V3_q1 (fifth 2) | F | 38,902 | +0.027 | +0.039 | +0.061 |
| V3_q1 (fifth 2) | L5 | 37,688 | +0.035 | +0.041 | +0.073 |
| V3_q2 (fifth 3) | F | 57,318 | +0.059 | +0.059 | +0.095 |
| V3_q2 (fifth 3) | L5 | 55,837 | +0.010 | +0.074 | +0.108 |
| V3_q3 (fifth 4) | F | 35,289 | +0.065 | +0.041 | +0.070 |
| V3_q3 (fifth 4) | L5 | 33,364 | +0.026 | +0.033 | +0.100 |
| V3_q4 (fifth 5) | F | 30,235 | +0.024 | +0.029 | +0.071 |
| V3_q4 (fifth 5) | L5 | 28,586 | +0.065 | +0.004 | +0.289 |

## Designs and counts

- Shot clusters (consecutive delivered shot numbers at most 3 apart): 86 with D1 discharges (89 over all files); size 1 to 31 discharges, median 4; largest five [24, 26, 28, 29, 31].
- V2 fold sizes (discharges): seed 1 [126, 129, 130, 124, 126], seed 2 [126, 126, 119, 125, 139], seed 3 [129, 132, 125, 124, 125].
- V3 fifths: 30801-31334 (127); 31335-31589 (127); 31590-31924 (127); 31925-32185 (127); 32188-32751 (127).
- V4: fit to 31991 412 discharges; fit to 32310 540; scored 95.
- D1 rows with the target defined: F 222,672, N1 214,768, L3 204,733, L5 215,369, L10 215,667.
- Rows without a complete input set (they keep G in every tree estimator): 2.62 % of the D1 rows.
- Readout run time 177 s.

## Remarks (facts, no verdicts)

- Under V4 the tree increment against L5 exceeds that against F (the opposite of V1). Concentration of TX over T0 over the scored discharges: V4a|F: 95 discharges, the five largest reductions hold 0.77 (shots [32522, 32748, 32312, 32328, 32493]), 32 discharges negative; V4a|L5: 83 discharges, the five largest reductions hold 0.45 (shots [32410, 32406, 32498, 32407, 32522]), 15 discharges negative; V4b|F: 95 discharges, the five largest reductions hold 0.57 (shots [32410, 32522, 32748, 32745, 32406]), 38 discharges negative; V4b|L5: 83 discharges, the five largest reductions hold 0.37 (shots [32406, 32522, 32527, 32410, 32408]), 14 discharges negative.
- The boosted trees respond to rounding-level changes of their inputs: between two implementations whose fast scalars differed by at most about 1e-7, TX over T0 on a re-partition moved by up to 0.002 (`b73_checks.md`).
- Of the 95 V4 test discharges, 83 have a row with L5 defined.

## Limits (registration section 9, unchanged)

- Exploratory on used data. The following level is built from reported values; a bias of the analysis that lasts 50 ms or more is in the target too. Shot clusters are a proxy for sessions. V3 confounds 'earlier only' with 'fewer fit discharges' for the early fifths; V4 is one block of 95 discharges. The network-based rows exist under V1 only; the gain λ of a fold is fitted on rows whose networks had this fold's discharges in training (one scalar). Whether the part that follows the fast channels is the fit, a displacement of the edge profile or the ion temperature is not decided by any readout here.
