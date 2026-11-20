# THESIS_RESULTS — experiment log (KSTAR CES nowcasting)

> **This file is the log, not the record in use.** Every analysis of the current line is recorded
> here in the order it was run, with its design, numbers and verdict. A number in this file is
> reference material until it is in `PROJECT_KNOWLEDGE.md` section 4; the paper and the next
> experiment take their numbers from there, and their definitions from `PAPER_DESIGN.md`.
>
> The log starts at B.70 (2026-11-19), the batch that fixed one data definition (D1) for every
> number. The entries before it (the windowed model, the recurrent network as the main estimator,
> the physics-filter family, the review loops of 2026-11-13 to 15, the batches B.6 to B.69) were
> removed in the cleanup of 2026-11-21 and are in the git history (last commit before the cleanup:
> `e86adaf`). They are not grounds for a next step.

## Entries

### B.70 phase 1: one data definition (D1) for every number of the paper -- every statement holds on every eligible row; the mixture rule flags 16 % of the rows, not the fit failures alone (2026-11-19)

Registration `PREREGISTRATION_B70.md` with addendum 1 (written after the timing and check runs, before any readout: re-scoring path, second-stage refit, mixture details, intervals, and a shared-reference scorer bit-identical to every arm's own script on r2 f0). CPU, nothing trained except the registered refit of the B.64 second stage; 135 stored networks of the 15 folds of r2 to r4 re-scored with `CES_MAX_VAL_SAMPLES=0`; run by an AI agent under the author's supervision. The machine slept from 19:48 to 21:47 and five unfinished folds were re-scored from the start. Output `docs/paper/b70_d1.{md,json}`; checks (a)-(d) pass (`docs/paper/b70_checks.md`): every key bit-identical except the network's error (CPU vs CUDA, at most 1.70 eV), row counts equal an independent recount, all arms of a fold on identical rows, the refitted second stage equal to the stored predictions (0.0 eV).

D1 rows: 222,672 per re-partition over 635 discharges (four never scored: s31002, s31331, s32501, s32570); largest 1,477 rows from one discharge, median 293; the old 40,000-sample subset was 0.862-0.863 of them.

| statement | value | verdict | expectation |
|---|---|---|---|
| D1-1 two-stage estimator over causal GP, MSE skill | +0.177 [+0.156, +0.196] | holds | holds, within 0.01 of B.64 (met) |
| D1-2 two-stage estimator over two-stage twin | +0.097 [+0.078, +0.116] | holds | holds (met) |
| D1-3 network over twin | +0.052 [+0.029, +0.074] | holds | holds (met) |
| D1-4 network over causal GP | +0.094 [+0.071, +0.114] | holds | holds (met) |
| D1-5 median absolute error, two-stage estimator / hold | 0.908 [0.887, 0.931] | holds | holds (met) |
| D1-6 D1-1 and D1-2 without the rows the mixture flags | +0.207 [+0.153, +0.250]; +0.065 [+0.010, +0.107] | holds | holds (met); about 1,000-2,000 flagged rows below 3 keV (not met) |

Reported: MSE (eV^2) two-stage 30,595 [27,336, 34,060], network 33,680, twin 35,543, causal GP 37,166, hold 54,120, offline GP 32,672; every main-text number moves by at most 0.008 in skill and 0.9 eV in RMSE. B.64 T1-T6 and the six B.59 statements keep their verdicts. B.68 X2 (auxiliary target over the twin) becomes +0.035 [-0.000, +0.063] (was +0.035 [+0.001, +0.062]).

Mixture (residual of ln T_i against the median of the 11 centred rows, EM on each fold's training discharges): bad fraction 0.1888-0.2086, good sd 0.0731-0.0772, bad sd 0.812-0.843; 35,340 scored rows flagged on average (0.1587), of them 11,467 single isolated frames and 4,211 below 75 eV; 1,012 of the 1,197 values above 3 keV flagged when run before the cut. Without the flagged rows the network alone is +0.040 [-0.045, +0.106] over the causal GP and -0.015 [-0.110, +0.052] over the twin; the two-stage estimator over the network +0.174 [+0.145, +0.201].

Reading. One definition and every eligible row leave every number of the paper within 0.01 and every verdict unchanged except B.68 X2; the paper takes its numbers from this readout. The mixture's good component is narrow (about 7.5 % in T_i), so every row that differs from its neighbours is flagged, whether a failed fit or a fast change; it reads T_i only and cannot tell the two apart. It is therefore not adopted as the outlier rule: the 3 keV rule stays, and D1-6 is reported as a sensitivity. On the unflagged rows the first stage's two gaps contain zero: its gain lives in the rows that differ from their neighbours, while the two-stage estimator keeps both gaps there.

### B.71: when the CES updates stop -- no divergence, but the error accumulates; the network is ahead of the causal GP only up to 50 ms; the return takes 0.1 to 1 s (2026-11-20)

Registration `PREREGISTRATION_B71.md` with addendum 1 (C1-C16, written before any readout: float64 forwards because the stepped and the batched LSTM differ by up to 3.7e-4 eV in float32 against the registered 1e-4 eV; a bin is outside the training range when its lower edge exceeds the 99.9th percentile of any fold; filter inputs from the CSV as in the filter's own loader). CPU, frozen D1 networks and twins of the 15 folds of r2 to r4; cuts every 1 s in each block (5,298 kept), 1,099,107 F/R units and 1,280,739 G units; run by an AI agent under the author's supervision; rows 42 min. Checks pass on every fold (`docs/paper/b71_checks.md`): rebuilt inputs equal build_blocks, U equals the B.70 re-scoring within 6.6e-4 eV on every row, a cut after the block end leaves F, R, G equal to U, stepped equals batched within 8.6e-13 eV. The session lead recomputed the MSE table of the network, hold and the causal GP from the row files; it agrees.

| statement | value | verdict | expectation |
|---|---|---|---|
| S1 network F over twin F, MSE reduction %, (100, 300] and (300, 1000] ms | +5.5 [-0.3, +12.7]; +12.3 [+2.0, +20.6] | does not hold | holds |
| S2 network F over hold F, every bin up to 1,000 ms | +50.8, +37.2, +23.8, +17.1 above zero; (300, 1000] +5.4 [-10.1, +18.6] | does not hold | holds |
| S3 network R divergent in no bin | MSE over hold F 0.49 to 1.84 [1.43, 2.36]; no estimate outside 0-3 keV | holds | holds (weak) |
| S4 network F over network R, (300, 1000] | +23.2 [+7.2, +37.4] | holds | holds |
| S5 returned by (30, 100] ms after gaps of 0.1, 0.3, 1 s | MSE(G)/MSE(U) 1.05 [1.01, 1.12], 1.36 [1.19, 1.61], 1.74 [1.32, 2.18] | does not hold | holds |

Reported: MSE (10^3 eV^2) by time since the last value, bins (0,10], (10,50], (50,100], (100,300], (300,1000], (1000,3000] ms: network U 17.89, 25.35, 28.18, 30.90, 32.39, 27.52; network F 17.89, 33.78, 55.66, 97.56, 122.87, 148.78; network R 17.89, 29.95, 42.99, 78.46, 161.11, 230.02; twin F 20.66, 39.73, 58.43, 103.35, 140.84, 152.31; hold F 36.34, 53.75, 73.05, 117.65, 129.93, 124.68; causal GP F 22.23, 39.07, 52.58, 91.30, 97.92, 93.99; KF0 F 23.05, 39.19, 56.29, 97.23, 104.20, 100.05; KFX F 20.83, 34.60, 51.00, 90.74, 116.36, 433.90. Network F over causal GP F: +19.5 [+3.5, +36.8], +13.5 [+5.1, +21.5], -5.9 [-17.1, +2.5], -6.9 [-13.0, -1.8], -25.5 [-42.9, -10.2], -58.3 [-108.7, -28.8]. Mean signed error (eV) in the last three bins: F -60.7, -123.6, -175.1; R +36.3, +126.2, +199.9. Return: after 0.1 s by (100, 300] ms (1.02 [1.00, 1.05]); after 0.3 s and 1 s by (300, 1000] ms. Staleness of the T_i input on the training rows: median 10 ms, 99th percentile 130-170 ms, 99.9th 450-540 ms, maximum 660-970 ms; (1000, 3000] is outside the training range in all 15 folds. KFX F has 0.0057 of its estimates outside 0-3 keV in the last bin.

Reading. Nothing diverges: fed its own estimate for 3 s the network stays in the physical range and below twice the MSE of hold. The error accumulates with a sign: low when told the true age, high when fed its own estimate. Beyond 50 ms without a CES value the network is not separated from the causal GP, and from 100 ms on the causal GP has the lower MSE: the estimator bridges gaps of the scale it was trained on (beam interruptions), and is not a replacement of CES over long outages; the paper says so. After the values return the network is back at the update level within 0.1 to 1 s, longer after longer gaps. Limits: no long outage in training; T_i only; one initialisation per fold; the two-stage estimator is not included. The measurement that would change this: an estimator trained with the CES history masked over random lengths (GPU).

### B.70 phase 2: the other cited readouts on D1 -- verdicts unchanged except B.66 R2 and B.65 I1, I3 (2026-11-20)

Registration `PREREGISTRATION_B70.md`, addendum 2 (C1-C10, written before any phase-2 readout: each batch's own readout code is run with its inputs redirected to the D1 rows; every pooled set becomes r2 to r4; checks that compared recorded old numbers compare D1 references; rows after an absent frame by the B.54 definition everywhere; the replay statements are read as pooled intervals). CPU, 3.3 h, run by an AI agent under the author's supervision; filters refitted per fold on the training discharges with held repeats removed (parameters in `data/.b70/kf_params_r{r}_f{k}.json`, used by B.71). Output `docs/paper/b70_d1_phase2.{md,json}`; checks in `docs/paper/b70_checks.md` (Phase 2) pass. The session lead recomputed the three delay-replay values from the row files (221,442 fixed rows): +0.111, +0.114, +0.087, equal.

| statement | old (treatment) | D1 | verdict |
|---|---|---|---|
| B.66 R2 increment over the twin in discharges with frequent bursts | +0.025 [-0.009, +0.052] (r2-r9) | +0.033 [+0.008, +0.061] | did not hold -> holds |
| B.65 I1 inclusive-trained network over causal GP, MSE | +0.109 [+0.070, +0.146] (scored on inclusive rows) | -0.092 [-0.168, -0.044] (scored on cut rows) | held -> does not hold |
| B.65 I3 the same, Huber | +0.037 [+0.003, +0.070] | -0.151 [-0.200, -0.106] | held -> does not hold |

Unchanged verdicts, D1 values: delay L = 1, 2 over the causal GP +0.111 [+0.090, +0.131], +0.114 [+0.094, +0.134]; L = 2 over the twin +0.087 [+0.066, +0.108]; one value per 50 ms over the causal GP +0.112 [+0.096, +0.129], over the twin +0.089 [+0.069, +0.109]; KFX over KF0 +0.049 [+0.040, +0.059], network over KFX +0.042 [+0.022, +0.060], KFX over the twin +0.011 [-0.006, +0.029]; all fast channels one frame old, over the twin -0.027 [-0.047, -0.010]; only BES one frame old +0.002 [-0.018, +0.023], only ECEI +0.031 [+0.010, +0.049], only Mirnov +0.052 [+0.029, +0.074]; rows passing the value rule +0.098 [+0.066, +0.126], flagged rows +0.015 [-0.010, +0.038]; thirds of T_i level over the twin +0.015 [-0.003, +0.035], +0.040 [+0.008, +0.072], +0.097 [+0.041, +0.140]; the 64 discharges with the largest reduction hold 0.7356 of it, and without them the network is +0.035 [+0.007, +0.057] over the causal GP and +0.026 [-0.006, +0.051] over the twin; the inclusive-trained network against the cut-trained one on the cut rows -0.205 [-0.280, -0.148].

Reading. The readouts the main text uses keep their verdicts on D1. B.66 R2 moves with the number of re-partitions and rows. B.65 changes because the population it is scored on changes: a network trained with the values above 3 keV kept is worse than the causal GP on the rows at or below 3 keV, so keeping those values in training damages the estimate elsewhere, which is the ground for applying the cut in training as well. Not re-read (training needed): the four splits, the time split, the cap-30 networks, B.53; they stay in the supplement with their treatment named.

### B.72: the D1 result under other validation conventions -- verdicts unchanged under four other inference methods; the fold-corrected interval is up to 1.63 times as wide (2026-11-20)

Registration `PREREGISTRATION_B72.md` with addendum 1. CPU re-read of the stored D1 estimates (968 s), run by an AI agent under the author's supervision; `arch` 9.0.0 for the model confidence set. Output `docs/paper/b72_methods.{md,json}`; its checks pass (the two-level bootstrap of B.70 reproduced draw for draw). The session lead recomputed the corrected t interval of the network over its twin from the fifteen fold skills and the R2 of hold and of the two-stage estimator from the MSEs; they agree.

| statement | value | verdict | expectation |
|---|---|---|---|
| V1 D1-1 to D1-4 keep their verdict under the discharge bootstrap, the cluster-robust paired t, the corrected resampled t over 15 folds and Wilcoxon | all keep it | holds | holds; uncertain for D1-3 under two of them |
| V2 the 95 % model confidence set of the causal estimators | the two-stage estimator alone | holds | holds |
| V3 Holm over the seven D1 statements | largest adjusted p 0.0246 | holds | holds |
| V4 median per-discharge MSE reduction of the two-stage estimator | +0.150 [+0.130, +0.170] over the causal GP, +0.072 [+0.058, +0.088] over its twin | holds | holds; weak over the twin |
| V5 no alternative interval more than 1.25 times as wide | corrected t: 1.46 (network over twin), 1.63 (network over causal GP), 1.20 and 1.07 for the two-stage statements | does not hold | uncertain for the corrected t |

Reported: R2 hold 0.6284, causal GP 0.7448, twin 0.7560, network 0.7688, two-stage twin 0.7674, two-stage estimator 0.7899, offline GP 0.7757, KFX 0.7585. Corrected t intervals: network over twin +0.053 [+0.020, +0.086], network over causal GP +0.094 [+0.059, +0.129], two-stage over causal GP +0.177 [+0.153, +0.201], over its twin +0.098 [+0.077, +0.118]. Share of discharges with the lower MSE: two-stage against its twin 0.719 [0.683, 0.754], network against twin 0.658 [0.620, 0.696].

Reading. Inside the random partition and the same-frame target, the inference method does not change a verdict. By the registered rule the network statements take the corrected t interval (PAPER_DESIGN.md 7). The batch does not vary the target or the validation design; four critiques of the same day show those to be the larger questions (next entry), and B.73 addresses them.

### Four critiques in the role of researchers, with why-chains (2026-11-20)

At the author's request four reviewers (a PPCF referee, a pedestal and control physicist, a statistician, a CXRS diagnostician) read the records, the code and the data without the session lead's conclusions or the earlier gradings, and wrote `docs/paper/review_loop/critique_20261120_{referee,pedestal,stats,cxrs}.md`, each with read-only checks. Their numbers are unregistered and after the fact. The session lead reproduced one with own code: scored against the next frame, the MSE reduction of the two-stage estimator over its twin keeps 0.24 [0.12, 0.37] of its same-frame size, the network's 0.27 [-0.08, 0.46], while the gain of the history-only second stage keeps 0.92 [0.79, 1.04] (205,337 rows per re-partition with regular neighbours).

Roots named by the reviewers: (1) the scored quantity is the reported value of the frame, not the level of the temperature; with the correction about halved the increment against the following level is about +0.06; (2) discharges are treated as exchangeable: the increment sits in shot clusters before shot 32000 and on the chronological split the network's increment over its twin is +0.005 [-0.019, +0.028] (the eight outer runs of B.69, read by the reviewer before that batch was finished; its registered statements were not computed), while boosted trees keep +0.02 to +0.04; (3) a causal GP with boosted trees on 18 scalars has the MSE of the two-stage estimator; (4) what the fast channels co-vary with is not attributed: the reported width follows the beam-emission light of its own exposure (log slope 1.28 [1.15, 1.44]) and the rotation from the same fit does not (0.018 [-0.016, +0.050]); where the BES sample falls the gain lasts (+0.114 [+0.074, +0.152], 6 % of frames), where it rises it does not (-0.013 [-0.049, +0.019], 29 %); (5) values are written for frames exposed without beam emission (13 of 15 in the raw streams), and the newest value the offline analysis could deliver is 0.65 s old at the median; (6) strata defined by the scored value (the reading of B.70 D1-6) support no statement. Accepted by the reviewers: the main table reproduces; no leakage in either stage; adjacent shots in training are worth 1 to 2 % of the squared error; the gain of the history-only estimators lasts into the following frames. Registered repetitions: B.73 (targets and validation designs, estimators without a network; addendum 1 with input-defined strata) and B.74 (bursts inside the exposure, raw streams).

### B.74: inside the exposure of a frame whose reported width departs -- a burst of beam-emission light; the following level and the rotation do not move (2026-11-20)

Registration `PREREGISTRATION_B74.md` with addendum 1 (written before any readout). CPU, the raw 500 kHz BES streams of the twelve discharges that have them (31921, the discovery discharge of B.31, reported apart; eleven in the statements), B.31's burst detector unchanged; no estimator; run by an AI agent under the author's supervision. Output `docs/paper/b74_exposure.{md,json}`; both checks pass (burst counts equal B.31's; the BES sample of a row equals the raw sample at the stamp). The session lead recomputed the medians of J2 and J3 and the burst counts with own code: 212 class-B rows with a ratio, median 1.105; 4,210 class-Q rows, 1.006; following-level ratio 0.994 and 1.004. Intervals resample the eleven discharges and are coarse.

Classes of a row for the window from 2 ms before to 8 ms after the stamp: B = a burst onset inside; Q = none inside and none in the 30 ms before. Ratio = value / median of the ten preceding rows.

| statement | value | verdict | expectation |
|---|---|---|---|
| J1 share of rows without a usable T_i, B minus Q | +0.101 [+0.017, +0.192] (0.144 against 0.044) | holds | holds |
| J2 median T_i ratio, B minus Q | +0.099 [+0.021, +0.187] (1.105 [1.030, 1.188] against 1.006 [1.001, 1.014]) | holds | uncertain |
| J3 median following-level ratio, B minus Q, not above zero | -0.014 [-0.096, +0.041] (0.990 [0.909, 1.042] against 1.004 [0.998, 1.007]) | holds | holds if J2 holds |
| J4 median fresh V_rot ratio, B minus Q, inside +-0.03 | +0.006 [-0.025, +0.025] (1.008 [0.963, 1.026] against 1.001 [0.989, 1.014]) | holds | holds |
| J5 the 5 ms window with the largest rank correlation starts between -2 and +5 ms | starts at -4 ms (0.451 [0.394, 0.504]; 0.361 [0.312, 0.411] at -10 ms, 0.287 [0.185, 0.372] at +10 ms) | does not hold | holds |
| J6 share of class B among upward one-frame departures over its share among all rows | 9.48 [4.65, 18.68] (0.419 [0.224, 0.654] against 0.044 [0.025, 0.067]) | holds | uncertain |

Reported: rows B 291, Q 4706. In burst frames the T_i ratio is above 1.35 in 0.259 [0.126, 0.372] of rows (Q: 0.042) and below 0.74 in 0.179 [0.080, 0.289] (Q: 0.045); the frame right after a burst reads 0.930 [0.802, 1.028]. Downward one-frame departures: share ratio 1.96 [0.63, 4.25]. The Mirnov RMS ratio (0.960) and the ECEI step across the burst onset (+0.002) equal those of class Q.

Reading. In these discharges a one-frame rise of the reported width goes with a burst of beam-emission light inside the exposure of that frame (13 of 31 upward departures); the level of the following frames and the rotation from the same fit do not move, and the value of a burst frame is scattered both ways or absent. This says in which frames the width departs, not whether the ions are hotter in that frame: no spectrum, fit amplitude or fit error exists here. J5 fails by location only: the light that goes with the value lies between 4 ms before the stamp and the stamp, so the BES sample "of the present frame" is taken after most of the frame's light, and the text does not say that the estimate precedes the measurement. Limits: eleven discharges of one campaign; bursts defined on the beam emission only; fresh rotation values in eight discharges.

### B.73: what the fast channels add, by target and by validation design, without a network -- an increment for the following level under random and cluster-blocked folds, not established forward in time; no network needed (2026-11-20)

Registration `PREREGISTRATION_B73.md`, addendum 1 (session lead, before any readout: strata defined on inputs, the rotation witness, rows without beam emission, Q11-Q13) and addendum 2 (the agent's implementation choices; its text first overwrote addendum 1 on disk and was moved by the session lead, addendum 1 restored from git). CPU, about 2 h, run by an AI agent under the author's supervision; boosted regressions with default settings, filters refitted per target; no network trained or run. Output `docs/paper/b73_targets.{md,json}`; checks `docs/paper/b73_checks.md` pass (the leakage check failed twice on bit-identity, not on causality: running sums crossing file boundaries, then non-deterministic rounding of a standard deviation; both fixed and every tree refitted before any readout). Session lead: every increment of the design table recomputed from the stored per-unit estimates with own aggregation (equal to four decimals); the L5 target recomputed from the CSV files on 531 rows (equal); T0[L5] of one unit refitted (skill over the causal GP +0.168 against +0.174, the fit rows differ by the agent's complete-input rule).

Targets: F = the reported value of the frame; L5 = the median of the measured values among the next five rows (at least three). Designs: V1 random discharge folds (r2-r4); V2 whole shot clusters held out (86 clusters, three assignments); V3 rolling origin (fifths by shot order, each of the later four fitted on earlier fifths only); V4 chronological (fit up to shot 31991, score from 32312, 95 discharges). T0 = causal GP plus boosted regression on ten history scalars; TX = with eight fast scalars; TE = with the four ECEI scalars; K0, KX the filters; each fitted to the target it is scored on. Intervals: shot clusters resampled.

| design | F: TX over T0 (KX over K0) | L5: TX over T0 (KX over K0) |
|---|---|---|
| V1 | +0.087 [+0.059, +0.122] (+0.050 [+0.032, +0.070]) | +0.075 [+0.039, +0.106] (+0.055 [+0.035, +0.080]) |
| V2 | +0.072 [+0.048, +0.102] (+0.048 [+0.031, +0.068]) | +0.057 [+0.034, +0.079] (+0.054 [+0.035, +0.078]) |
| V3 | +0.045 [+0.023, +0.072] (+0.041 [+0.023, +0.060]) | +0.031 [-0.006, +0.060] (+0.042 [+0.021, +0.067]) |
| V4 | +0.019 [+0.001, +0.036] (+0.027 [+0.006, +0.050]) | +0.084 [+0.048, +0.123] (-0.007 [-0.038, +0.010]) |

| statement | value | verdict | expectation |
|---|---|---|---|
| Q1 as built, two-stage: MSE reduction against L5 below half of that against F | ratio +0.116 [-0.267, +0.403] | holds | holds |
| Q2 V1, L5 | +0.075 [+0.039, +0.106] | holds | holds |
| Q3 V2, L5 | +0.057 [+0.034, +0.079] | holds | holds |
| Q4 V3, L5 | +0.031 [-0.006, +0.060] | does not hold | uncertain |
| Q5 V4, L5 | +0.084 [+0.048, +0.123] | holds | uncertain |
| Q6 V1, L5, ECEI only over T0 | +0.032 [+0.022, +0.043] | holds | uncertain |
| Q7 V1, L5, TX over the gain-fitted two-stage estimator | +0.065 [+0.031, +0.095] | holds (above zero) | holds |
| Q8 V4, L5, T0 over the causal GP | +0.254 [+0.214, +0.307] | holds | holds |
| Q9 increment against F exceeds that against L5 (V1; V4) | +0.025 [-0.012, +0.064]; -0.065 [-0.115, -0.017] | neither holds | holds; uncertain |
| Q10 V4, AUC of the usability classifier with the fast scalars minus without | +0.023 [+0.012, +0.036] | holds | holds |
| Q11 rows with the BES sample above 1.03 of its recent level: above zero against F and not against L5 | F +0.099 [+0.068, +0.133]; L5 +0.065 [+0.027, +0.098] | does not hold | holds |
| Q12 rows with the BES sample below 0.9: L5 above zero | +0.183 [+0.124, +0.254] | holds | holds |
| Q13 log-log slope on the BES ratio: T_i above 0.5, fresh V_rot within +-0.1 | +1.285 [+1.147, +1.422]; +0.024 [-0.009, +0.058] | holds | holds |

Reported: as built against L5, two-stage over its twin +0.013 [-0.025, +0.045], network over twin +0.016 [-0.036, +0.054]; with the gain fitted (lambda 0.864 for F, 0.529 for L5; network 0.695 and 0.536) +0.062 [+0.046, +0.077]. RMSE against L5 (eV), V1: hold 194.7, causal GP 145.6, T0 132.9, TX 127.8, two-stage as built 135.6, gain-fitted 132.2; V4: hold 248.7, causal GP 166.6, T0 143.9, TX 137.7, K0 139.1, KX 139.7. T0 over the causal GP against L5: +0.166 (V1), +0.150 (V2), +0.134 (V3), +0.254 (V4). TX over the two-stage estimator as built against F +0.001 [-0.013, +0.020]. BES only over T0, V1, L5 +0.065 [+0.035, +0.093]; ECEI only, V4, L5 +0.058 [+0.030, +0.082]. Filter gains fitted to F: b_c 0.501, b_g 0.454; to L5: 0.525, 0.182. Usability classifier: AUC 0.9285 to 0.9622 under V1 (+0.034 [+0.026, +0.043]), 0.9242 to 0.9477 under V4. In the class above 1.03 the two-stage estimator as built has -0.064 [-0.133, -0.006] against L5. Rows of discharges without any absent frame (0.048 of rows): TX over T0 against F +0.016 [-0.021, +0.055]. Under V4 the five discharges with the largest reduction hold 0.45 of the L5 reduction of TX over T0.

Reading. (1) Estimators trained on the reported value of the frame carry no fast-channel gain into the level of the following frames; with the size of the correction fitted to that target the gain is +0.06. (2) Fitted to the level, the fast channels add +0.075 under random folds and +0.057 with whole shot clusters held out, and the increment against the frame is not established as larger. (3) Forward in time it is not established: under rolling origin the trees' interval contains zero while the filter pair's does not, and on the last block the trees and the filters disagree (+0.084 against -0.007; the best history-only RMSE there is the filter's). (4) What the history-only learned estimator adds over the causal GP holds under every design (+0.13 to +0.25 against the level). (5) No recurrent network is needed: the trees equal the two-stage estimator against the frame and are ahead of its gain-fitted form against the level. (6) ECEI alone, which uses neither the beam nor the spectrometer, gives part of the level increment. By the rules registered beforehand the paper names two targets, prints the whole design table, takes the estimator without a network, and states the increment forward in time as not established. Limits: no untouched data; the level target is the same automatic analysis; V4 has 19 clusters; a network trained on the level was not run (GPU).

### B.75: the estimate when the CES history is as old as it really is -- with a 0.65 s old history the fast channels reduce the MSE by 19 to 34 % under all four validation designs (2026-11-20)

Registration `PREREGISTRATION_B75.md` with addendum 1 (implementation choices, written before any readout). CPU, the network-free estimators of B.73 refitted per availability rule of the CES history; run by an AI agent under the author's supervision. Output `docs/paper/b75_availability.{md,json}`, checks `docs/paper/b75_checks.md` (five checks pass: the causal GP under D10 equals the cached one bit for bit; T0 and TX8 under D10 reproduce B.73's stored estimates with deviation 0.0 eV; 400 rows unchanged bit for bit when every value later than their visibility limit is removed or raised; the ages under the schedule rule reproduce the reviewer's C12; the units are B.73's). 1,566 fits.

Rules: D10 (every earlier frame visible, the condition of all earlier batches), D50 to D1000 (values at least d old), S (values before the most recent beam-off frame; zero analysis time assumed). Target L5 (the following level). T0 = causal GP on the visible values plus boosted trees on ten history scalars; TX = T0 plus eight fast scalars and six anchor scalars (fast level at the newest visible value, its change since, log ratio of raw levels); O = the frame's own reading. Fixed-delay rules fitted and scored on the rows complete under all six (160,343 rows, 562 discharges); S on 147,837 rows. Intervals: shot clusters resampled.

Increment TX over T0, target L5:

| design | D10 | D650 | S |
|---|---|---|---|
| V1 random | +0.103 [+0.070, +0.141] | +0.336 [+0.285, +0.380] | +0.327 [+0.262, +0.388] |
| V2 cluster-blocked | +0.058 [+0.019, +0.088] | +0.282 [+0.215, +0.343] | +0.293 [+0.216, +0.364] |
| V3 rolling origin | +0.028 [-0.011, +0.067] | +0.190 [+0.033, +0.292] | +0.231 [+0.146, +0.323] |
| V4 fit to 31991 | +0.089 [+0.032, +0.136] | +0.237 [+0.047, +0.415] | +0.028 [-0.166, +0.172] |
| V4 fit to 32310 | +0.079 [+0.058, +0.113] | +0.216 [+0.075, +0.360] | -0.254 [-0.541, +0.039] |

| statement | value | verdict | expectation |
|---|---|---|---|
| R1 V1: MSE(T0 under D650) / MSE(T0 under D10) above 1.5 | 2.958 [2.491, 3.484] | holds | holds (2 to 6) |
| R2 V1, D650: increment above zero | +0.336 [+0.285, +0.380] | holds | holds (+0.05 to +0.25); point above the range |
| R3 V1: increment under D650 exceeds that under D10 | +0.233 [+0.180, +0.281] | holds | uncertain, leaning to holds |
| R4 V2, D650 | +0.282 [+0.215, +0.343] | holds | holds |
| R5 V3, D650 | +0.190 [+0.033, +0.292] | holds | uncertain |
| R6 V4 (fit to 31991), D650 | +0.237 [+0.047, +0.415] | holds | uncertain |
| R7 V1, S | +0.327 [+0.262, +0.388] | holds | holds |
| R8 V4 (fit to 31991), S | +0.028 [-0.166, +0.172] | does not hold | uncertain |
| R9 V1, D650: TE over T0 above zero | +0.229 [+0.162, +0.291] | holds | uncertain (+0.02 to +0.10); point above the range |
| R10 V1 and V4, D650: T0 over G above zero under each | V1 +0.336 [+0.209, +0.420]; V4 +0.012 [-0.279, +0.164] | does not hold | holds |
| R11 V1 and V4, D10: MSE of TX below that of O | V1 +0.538 [+0.484, +0.585]; V4 +0.682 [+0.635, +0.741] | holds | holds (+0.3 to +0.6) |
| R12 V1, S: MSE of TX below that of O | -0.003 [-0.214, +0.163] | does not hold | does not hold |
| R13 D10, fifths 2 and 3: increment of V3 minus that of the size-matched control below zero | -0.036 [-0.069, +0.001] | does not hold | uncertain |

RMSE (eV), V1, target L5, by rule (hold of the newest visible value / causal GP / T0 / TX; O 183.3 on the common set, 178.2 on the S set): D10 193.5 / 144.1 / 131.5 / 124.5; D100 243.3 / 190.1 / 165.8 / 144.7; D300 293.8 / 240.7 / 200.8 / 162.5; D650 325.9 / 277.6 / 226.2 / 184.3; D1000 345.9 / 303.2 / 242.1 / 193.1; S 274.7 / 226.3 / 217.6 / 178.5.

Reported: BES half alone under V1, D650 +0.246 [+0.200, +0.292]; under V4 (fit to 31991), D650, ECEI half +0.177 [+0.084, +0.304], BES half -0.014 [-0.257, +0.203]. Without the anchor scalars (TX8): V1, D650 +0.168 [+0.128, +0.203]; V4, D650 -0.050 [-0.544, +0.314]. Recovered share of the staleness loss: V1 D650 0.508 [0.442, 0.569], S 0.509 [0.421, 0.591]; V3 D650 0.277 [0.051, 0.415]; V4 D650 0.487 [0.101, 0.817]. The D10 fit applied to D650 inputs without refit: RMSE T0 270.3, TX 259.7 (refitted 226.2, 184.3; refitted TX over applied +0.496 [+0.392, +0.573]). Rolling origin by fifth, D650: +0.266, -0.212 [-0.527, +0.157], +0.336, +0.254. TX against the frame's own reading under S: V2 -0.110 [-0.340, +0.072], V3 -0.183 [-0.476, +0.073], V4 +0.457 [+0.150, +0.601]. Computation per row: median 1.95 ms, 99th percentile 4.59 ms (one thread, shared machine).

Session lead's checks (`docs/paper/b75_lead_checks.md`, code written independently): the table's increments and R9, R11, R12 recomputed from the stored estimates agree; the D650 inputs of all D1 rows (visible values, age, history scalars, causal GP, anchor row) recomputed bit for bit, the fast inputs up to rounding (largest difference 5e-7 standard units), and that rounding alone moves the V4 (fit to 31991), D650 increment from +0.237 to +0.217. Unregistered controls without intervals: fast inputs permuted over rows and refitted +0.013; fast scalars taken at the anchor row instead of the present row -0.128 (present ones on the same rows +0.205); the time since the last beam interruption alone adds +0.023 to T0, and with it in both estimators the V1 (r2), D650 increment is +0.335 (without +0.336); on rows more than 100 ms after an interruption +0.320.

Reading. With the previous frame visible the fast channels add little and nothing established forward in time (B.73). With a history 0.65 s old they reduce the MSE of the level estimate by 19 to 34 % under every design, half of what staleness costs under the random design; the gain comes from the change of the fast level since the newest visible value and holds for ECEI alone. Under the beam-modulation schedule it holds under the random, cluster-blocked and rolling-origin designs and is not established on the last block (50 discharges). An estimator fitted with a fresh history does not carry over to a stale one; this is the condition B.71 measured. By the rule registered beforehand (R4 and R5 hold) the word nowcasting may stay in title and abstract; the decision is the author's. Limits: exploratory on used data, one campaign, the history values are the offline product, zero analysis time under S, one estimator per rule.
### Prior work on missing T_i and a novelty critique (2026-11-20, unregistered)

The author asked whether work that fills in missing charge-exchange values really does not exist, and judged that at the present level the work has nothing that prior work lacks. Two records: `docs/paper/prior_work_missing_ti.md` (the session lead's search; RTCAKENN read in full text) and `docs/paper/review_loop/critique_20261120_novelty.md` (a referee-role agent, with its search log and three unregistered checks N1 to N3).

- Estimating T_i without the charge-exchange measurement exists (RTCAKENN on DIII-D by input dropout, no time history, no accuracy given for the case without CER, edge target extrapolated from T_e; an EAST preprint). Not found: a causal estimate in time of one chord against hold or a GP with intervals, anything for KSTAR CES, and the error as a function of the age of the newest value.
- The critique's verdicts on the findings that existed before B.75: the history-only learned estimator over the causal GP follows from known results and rests on a weak reference (a four-coefficient linear blend gives +0.086 [+0.072, +0.101], T0 +0.168 [+0.148, +0.187]; under V4 the five-parameter filter has RMSE 139.1 against 143.9 eV); the burst finding is the known ELM disturbance of edge CXRS; the protocol exists component by component; the decomposition of the cross-diagnostic gain by target is the one item not found in the fusion literature. For PPCF: nothing; after a second demonstration on MAST, a methods paper.
- The author's direction of the same day: a paper proposes a possibility and shows that it works; every candidate is to be tried (A: B.77, B: B.75 and B.76, D: B.78; C needs a GPU for the network form).

### B.77: the frame's own reading corrected and flagged with the fast samples of the same frame -- the fast samples add +0.048 under the random design and nothing established under the other three (candidate A, 2026-11-20)

Registration `PREREGISTRATION_B77.md` with addendum 1 (written before any fit). CPU; run by an AI agent under the author's supervision. Output `docs/paper/b77_correct.{md,json}`, checks `docs/paper/b77_checks.md`, the session lead's checks `docs/paper/b77_lead_checks.md` (statement values recomputed from the stored estimates; C0 refitted with own code bit for bit in two units). Rows: 209,432 (606 discharges, 85 clusters), rule D10, target L5. O = the reported value of frame t; C0 = causal GP plus boosted trees on ten history scalars and four scalars of the frame's value; CX = C0 plus fourteen fast scalars.

RMSE (eV) O / MED3 / T0 / TX / C0 / CX: V1 183.4 / 145.1 / 129.9 / 124.2 / 120.0 / 117.1; V2 183.4 / 145.1 / 131.4 / 127.6 / 121.1 / 120.2; V3 203.3 / 158.6 / 146.1 / 143.9 / 134.5 / 135.1; V4 (fit to 31991) 234.7 / 155.7 / 133.9 / 128.2 / 120.7 / 120.1; V4 (fit to 32310) 234.7 / 155.7 / 126.6 / 123.6 / 117.1 / 114.5.

| statement | value | verdict | expectation |
|---|---|---|---|
| A1 V1: C0 below O | +0.572 [+0.518, +0.621] | holds | holds (+0.5 to +0.7) |
| A2 V1: C0 below T0 | +0.147 [+0.132, +0.164] | holds | holds (+0.05 to +0.30) |
| A3 V1: CX over C0 above zero | +0.048 [+0.023, +0.072] | holds | holds (+0.03 to +0.15) |
| A4 V2: CX over C0 | +0.013 [-0.012, +0.037] | does not hold | holds |
| A5 V3: CX over C0 | -0.009 [-0.049, +0.023] | does not hold | uncertain |
| A6 V4 (fit to 31991): CX over C0 | +0.010 [-0.009, +0.032] | does not hold | uncertain |
| A7 V1: CX over C0 exceeds TX over T0 | -0.038 [-0.054, -0.018] | does not hold | uncertain, leaning to holds |
| A8 V1 and V4: AUC of FHX minus FH above zero under each | V1 +0.005 [+0.003, +0.008]; V4 -0.000 [-0.007, +0.007] | does not hold | holds |
| A9 V1, 10 % screened: MSE of O kept by FHX over kept by FH below 1 | 0.989 [0.975, 1.002] | does not hold | uncertain |
| A10 V1: CE over C0 above zero | +0.014 [+0.003, +0.025] | holds | uncertain |
| A11 V1, placebo | -0.004 [-0.008, +0.000] | holds | holds |

Reported: V4 (fit to 32310) CX over C0 +0.043 [+0.033, +0.062]; by BES class, below 0.9 of the recent level +0.100 [+0.047, +0.164], 0.97 to 1.03 +0.024 [-0.004, +0.047]; areas under the curve PH / PHX / FH / FHX: V1 0.8707 / 0.8864 / 0.9171 / 0.9223, V4 0.7685 / 0.7693 / 0.8763 / 0.8759; PHX minus PH: V1 +0.016 [+0.012, +0.021], V4 +0.001 [-0.009, +0.015]; screening under V1, MSE of O 33,642 eV^2 unscreened and with 10 % dropped FHX 14,513, FH 14,680, PHX 23,753, PH 24,816; fast scalars of the row before: +0.035 [+0.014, +0.055].

Reading. A value corrected from the frame's reading and the history is far closer to the following level than the reading as it is, which is expected of any smoother. What the fast samples add to that correction holds under the random design only; under the cluster-blocked, rolling-origin and chronological (fit to 31991) designs the interval contains zero, and their part in the flags is small or not established. By the rule registered beforehand (A3 and A4 together) candidate A is not proposed. Once the frame's value is read, most of what the fast samples announced about that frame is no longer needed (A7).

### B.78: a CES value every k-th frame -- with a value every 100 ms the fast channels reduce the MSE of the level estimate by 19 % (random), 16 % (cluster-blocked), 8 % (rolling origin); not established on the last block (candidate D, 2026-11-20)

Registration `PREREGISTRATION_B78.md` with addendum 1 (written before any readout). CPU; run by an AI agent under the author's supervision. Output `docs/paper/b78_sparse.{md,json}`, checks `docs/paper/b78_checks.md` (frame lattice; the K1 inputs equal B.75's D10 inputs bit for bit on 88,749 rows; 200 rows unchanged when values off the kept grid are removed or raised), the session lead's checks `docs/paper/b78_lead_checks.md` (statement values recomputed; frame index, scored rows and K10 history inputs recomputed with own code bit for bit). Scored rows: frames on no kept grid, complete under all six rules, 60,523 rows (507 discharges, 72 clusters). Estimators and inputs as B.75; target L5; intervals with shot clusters resampled.

RMSE (eV), V1, T0 / TX by rule: K1 127.0 / 120.9; K2 130.2 / 123.8; K5 145.6 / 135.4; K10 160.7 / 144.6; K20 179.7 / 159.9; K50 212.2 / 180.1; the frame's own reading 177.1.

| statement | value | verdict | expectation |
|---|---|---|---|
| K-1 V1, K10: TX over T0 above zero | +0.191 [+0.133, +0.247] | holds | holds (+0.10 to +0.30) |
| K-2 V2, K10 | +0.163 [+0.095, +0.231] | holds | holds |
| K-3 V3, K10 | +0.075 [+0.012, +0.123] | holds | uncertain, leaning to holds |
| K-4 V4 (fit to 31991), K10 | +0.023 [-0.077, +0.120] | does not hold | uncertain |
| K-5 V1: MSE of TX under K10 below that of T0 under K5 | +0.015 [-0.046, +0.078] | does not hold | uncertain |
| K-6 V1: MSE(TX, K10) / MSE(TX, K1) below 2 | 1.430 [1.332, 1.530] | holds | holds (1.3 to 2.0) |
| K-7 V1, K10: TE over T0 above zero | +0.107 [+0.052, +0.161] | holds | holds |
| K-8 V1, K10, placebo | -0.003 [-0.015, +0.009] | holds | holds |

Reported: increments by rule, V1 +0.093, +0.096, +0.136, +0.191, +0.208, +0.279 (K1 to K50); V4 (fit to 32310), K10 +0.105 [+0.018, +0.201]; equivalent rate under V1, the spacing at which T0 has the MSE of TX: K5 2.69 [2.05, 3.25] frames, K10 4.67 [3.59, 5.78], K20 9.64 [7.53, 11.86], K50 20.20 [15.41, 24.71]; recovered share K5 0.564 [0.378, 0.848], K10 0.508 [0.376, 0.647]; under V4 (fit to 31991), K10, ECEI half +0.075 [+0.037, +0.128], BES half -0.134 [-0.298, +0.032]; by position between two kept values (V1, K10) +0.122 [+0.089, +0.159] at 10 to 30 ms and +0.204 [+0.146, +0.263] at 70 to 90 ms; fast scalars taken at the anchor row +0.078 [+0.038, +0.116] over T0.

Reading. As with a delayed history (B.75), the sparser the CES values the more the fast channels add. With a value every 100 ms the increment holds under the random, cluster-blocked and rolling-origin designs and is not established on the chronological split (3,587 scored rows). With the fast channels a value every 100 ms gives the error of a value about every 47 ms without them. By the rule registered beforehand (K-1 and K-2 hold) candidate D may be proposed with these sizes. Limit: the thinning is done after the fact; the withheld frames were taken with the beam on.

### B.76: MAST, a stale charge-exchange history with Thomson scattering of the present at the chord radius -- with a history 100 ms old the Thomson scalars reduce the MSE of the level estimate by 28 to 44 % under all four designs (candidate B on a second device, 2026-11-20)

Registration `PREREGISTRATION_B76.md` with addendum 1 (sections 1 to 9 before any run). CPU; run by an AI agent under the author's supervision. Output `docs/paper/b76_mast_availability.{md,json}`, checks `docs/paper/b76_checks.md` (H under D5 equals the last visible value of the earlier MAST row tables on 161,461 matched rows; 200 rows unchanged bit for bit under removed and raised later values; sessions whole and fits before the first test shot; Thomson windows), the session lead's checks `docs/paper/b76_lead_checks.md` (24 shots recomputed from the shot files with own code, exact; own fits of the chronological design, exact; unregistered controls). Rows: pedestal-top chords (1.5 to 5 cm inside the Thomson density edge), no frame hidden, complete under all six rules: 81,360 rows, 1,785 shots, 224 sessions (session = experiment date). Target L5 = median of the good values of the next five frames. T0 = causal GP plus boosted trees on history scalars and normalised radius; TX = T0 plus nine Thomson scalars (T_e, n_e at frame t and at the anchor frame, differences, log ratios, distance to the density edge). Intervals: sessions resampled.

Increment TX over T0, target L5 (D5 / D25 / D50 / D100 / D150): M1 random +0.107 / +0.224 / +0.258 / +0.305 / +0.283; M2 session-blocked +0.082 / +0.217 / +0.241 / +0.288 / +0.275; M3 rolling origin +0.035 / +0.192 / +0.138 / +0.277 / +0.164; M4 chronological +0.226 / +0.334 / +0.471 / +0.443 / +0.472.

| statement | value | verdict | expectation |
|---|---|---|---|
| S1 M1: MSE(T0 under D100) / MSE(T0 under D5) above 1.5 | 1.567 [1.427, 1.745] | does not hold | holds (2 to 5) |
| S2 M1, D100: increment above zero | +0.305 [+0.247, +0.371] | holds | holds (+0.05 to +0.30) |
| S3 M1: increment under D100 exceeds that under D5 | +0.198 [+0.144, +0.265] | holds | holds |
| S4 M2, D100 | +0.288 [+0.219, +0.366] | holds | holds |
| S5 M3, D100 | +0.277 [+0.195, +0.358] | holds | uncertain, leaning to holds |
| S6 M4, D100 | +0.443 [+0.248, +0.610] | holds | uncertain |
| S7 M1, D100: RTa below H | -0.487 [-1.211, -0.007] | does not hold | uncertain |
| S8 M1, D100: TD over T0 above zero | +0.147 [+0.099, +0.199] | holds | uncertain |
| S9 M1, D100, placebo | +0.003 [-0.004, +0.011] | holds | holds |
| S10 M1, D5: TX below the frame's own reading | +0.346 [+0.252, +0.438] | holds | uncertain |

RMSE (eV), H / G / T0 / TX / O: M1 D5 219.1 / 187.3 / 172.0 / 162.6 / 201.1; M1 D100 308.8 / 265.3 / 215.3 / 179.5; M4 D5 123.6 / 108.0 / 92.1 / 81.0 / 90.0; M4 D100 244.0 / 167.1 / 117.9 / 88.0. Recovered share under D100: M1 0.844 [0.710, 0.998], M2 0.772, M3 0.578. TX under D100 against T0 under D5: -0.089 [-0.188, -0.001]. Thomson of the anchor frame only: +0.058 [+0.028, +0.089]; the present Thomson over that +0.262 [+0.210, +0.320]. Closed form RT = H times the T_e ratio: RMSE 677.8 (D5) to 1879.9 eV (D150) under M1; alpha of RTa 0.180 (D5) to 0.519 (D100). Target F, D100: M1 +0.189, M2 +0.173, M3 +0.131, M4 +0.255. Lead's unregistered controls on M4, D100: Thomson scalars permuted -0.008; T_e scalars only +0.433 of +0.443; n_e scalars only +0.170; edge distance only -0.013.

Reading. On a second device, with an electron diagnostic of another light source and instrument, the direction of B.75 is found again and holds forward in time: the present electron temperature at the chord radius carries the level of the ion temperature across a stale charge-exchange history. The plain ratio form does not work (two Thomson values divided are too noisy); the regression is needed. Nothing from MAST supports a KSTAR number; MAST carries the statement that the direction is not specific to one device and one pair of diagnostics. Limits: exploratory on used data; the delays are imposed by us; the pedestal-top band comes from the same Thomson profile that supplies the inputs.

### The NBI files (delivered 2026-11-20) and the registration of B.79

`data/nbi/s{shot}_NBI.csv` (884 files; all 641 shot files in use have one; same rows and time stamps in all 641). File facts in `docs/paper/nbi_files.md` from `ces_prediction/analyze_nbi_files.py` (no estimator, no scoring): NB1A never runs; NB1B is on in 0.987 of the rows (91.0 kV), NB1C 0.721 (59.4 kV), NB2A 0.882, NB2B 0.729, NB2C 0.306. The first row after an absent frame has NB1B at 0.614 and NB1C at 0.609 of their level and the NBI-2 sources at 1.000: an absent frame is a notch of the two NBI-1 sources, and that row's power is a sample of the returning beam. CES values are written with either NBI-1 source on (1B only 0.917 of 16,387 rows, 1C only 0.965 of 1,177 rows). 19 shots (149 rows, 145 with a T_i value) have no source above 0.1 MW.

Registered the same day: B.79 (`PREREGISTRATION_B79.md`), the beam waveforms as inputs of the estimators of B.75 (19 beam scalars: NBI-1, NBI-2 and total power at the present and at the anchor row with their differences, running means of the total, which sources are on, the voltage of NB1B). The question: what the fast diagnostics add beyond what the actuators already say, and whether the beam inputs change the last block under the schedule rule.

### B.79: the beam waveforms as inputs -- with the beam waveforms given, the fast diagnostics still reduce the MSE of the level estimate by 25 to 30 % under all four designs when the CES history is 0.65 s old (2026-11-20)

Registration `PREREGISTRATION_B79.md` with addendum 1. CPU; run by an AI agent under the author's supervision. Output `docs/paper/b79_beam_inputs.{md,json}`, checks `docs/paper/b79_checks.md` (every grid row joined to its beam row, largest stamp difference 9.5e-7 s; T0 and TX reproduce B.75 bit for bit; no beam scalar of a row uses a later row), the session lead's checks `docs/paper/b79_lead_checks.md` (statement values recomputed; the 19 beam scalars rebuilt from the NBI files with own code, exact; own fits of T0 and TN on the chronological design, exact). Rows, target L5, designs and rules as B.75. TN = history plus 19 beam scalars; TX = history plus 14 fast scalars; TXN = both. Intervals: shot clusters resampled.

| statement | value | verdict | expectation |
|---|---|---|---|
| N1 V1, D650: TN over T0 above zero | +0.148 [+0.086, +0.199] | holds | holds (+0.03 to +0.15) |
| N2 V1, D650: TXN over TN above zero | +0.258 [+0.205, +0.304] | holds | holds (+0.20 to +0.35) |
| N3 V2, D650: TXN over TN | +0.298 [+0.206, +0.378] | holds | holds |
| N4 V3, D650: TXN over TN | +0.263 [+0.162, +0.363] | holds | holds |
| N5 V4 (fit to 31991), D650: TXN over TN | +0.245 [+0.061, +0.357] | holds | uncertain, leaning to holds |
| N6 V1, S: TXN over TN | +0.253 [+0.184, +0.319] | holds | holds |
| N7 V4 (fit to 31991), S: TXN over TN | +0.102 [-0.156, +0.242] | does not hold | uncertain |
| N8 V4 (fit to 31991), S: TN over T0 | -0.100 [-0.282, +0.119] | does not hold | uncertain |
| N9 V1, D650: TXN over TX above zero | +0.047 [+0.023, +0.070] | holds | uncertain (0.00 to +0.05) |
| N10 V1, D10: TXN over TN | +0.097 [+0.063, +0.144] | holds | holds (+0.05 to +0.10) |
| N11 V1, D650, beam scalars permuted | -0.003 [-0.008, +0.001] | holds | holds |

Reported: TN over T0 under D650 outside the random design: V2 -0.044 [-0.207, +0.095], V3 -0.126 [-0.370, +0.047], V4 -0.066 [-0.497, +0.326]; TXN over TX there: -0.020 [-0.080, +0.021], -0.025 [-0.098, +0.030], -0.054 [-0.132, +0.034]. RMSE (eV), V1, T0 / TN / TX / TXN: D650 226.2 / 208.8 / 184.3 / 179.9; S 217.6 / 202.6 / 178.5 / 175.1. Under S, TXN over TN: V2 +0.238 [+0.157, +0.318], V3 +0.175 [+0.100, +0.252]. Ratio (TXN over TN) / (TX over T0): V1 D650 0.766 [0.657, 0.878], S 0.772 [0.613, 0.902]; V2 D650 1.057 [0.745, 1.485]. Halves over TN (V1, D650): ECEI +0.170 [+0.110, +0.228], BES +0.181 [+0.138, +0.227]. By time since the last notch (V1, D650): within 100 ms TXN over TN +0.305 [+0.230, +0.375] and TN over T0 +0.262 [+0.119, +0.385]; later rows +0.254 [+0.196, +0.309] and +0.103 [+0.034, +0.162]. V1 by block under S, TXN over TN: +0.269 [+0.192, +0.344], +0.213 [+0.050, +0.329], +0.086 [-0.136, +0.231]. Fast scalars permuted: +0.001 [-0.006, +0.009].

Sensitivity, judged on nothing: refitted with the 13 files whose NBI-2 power columns are invalid (`docs/paper/nbi_files.md` section 5) left out of every fit and score, N1 +0.149, N2 +0.248, N3 +0.299, N4 +0.246, N5 +0.249, N6 +0.250, N7 -0.070 [-0.437, +0.076], N8 +0.091 [-0.084, +0.236], N9 +0.040, N10 +0.093; every verdict would be the same. Order of events: the agent reported the files before any fit and named the registration's literal reading as its default; the lead's instruction to leave them out was sent before the fits and reached the agent after the fits and the readout; the registered statements are judged on the readout with the columns as delivered.

Reading. The increment of the fast diagnostics under a stale history is not what the beam waveforms already give: with the waveforms given it is 25 to 30 % under every design. The beam waveforms alone add under the random design only. The last block under the schedule rule is not established with beam inputs either. By the rules registered beforehand the paper gives both increments of the fast diagnostics and names the forward-in-time designs; N9 holds under the random design, so by the rule the estimator would include the beam scalars, with the note that under the other three designs their addition contains zero.

### The last block under the schedule rule: an unregistered diagnostic, and the registration of B.80 (2026-11-20)

`docs/paper/b75_last_block.md` (session lead; stored estimates of B.75 under V1 and rule S, input-only quantities, no interval). The age of the newest visible value is the same in the three blocks of shots (median 0.66, 0.60, 0.66 s). What differs from 32312 on is the ECEI: the change of its level since the newest visible value does not go with the change of the ion-temperature level (Spearman -0.013 against +0.269 and +0.329; BES +0.166 against +0.287 and +0.301); at a beam notch the ECEI level stays (1.001 of the level before, against 0.902 and 0.967) while T_i (0.757, 0.819, 0.837) and BES (0.669, 0.656, 0.643) fall in all blocks; its four channels are correlated at 0.654 against 0.911 and 0.841. Level of the history and state of NBI-2 do not remove the difference. Three discharges carry most of the loss (skill +0.133 without them, +0.066 with them). The cause is not decided: no toroidal field and no ECEI setting is in the files. B.80 (`PREREGISTRATION_B80.md`) asks whether the coherence of the ECEI channels orders the increment discharge by discharge inside every block, and reads an estimator that uses the fast inputs only where the channels are coherent.

### B.80: the coherence of the ECEI channels does not account for the limit under the schedule rule (2026-11-20)

Registration `PREREGISTRATION_B80.md` with addendum 1 (written before the check and any readout). CPU, no fit: stored estimates of B.75 and B.79 read by classes of an input-only measure per discharge, C_E = the mean pairwise correlation of the four raw ECEI channels (low below 0.7, middle, high from 0.9), and a gated estimator TG (TX where C_E is 0.7 or above, T0 otherwise). Run by an AI agent under the author's supervision. Output `docs/paper/b80_ecei_gate.{md,json}`; the session lead's checks `docs/paper/b80_lead_checks.md` (C_E, the class counts, the class skills and TG recomputed with own code). Target L5; intervals with shot clusters resampled.

C_E over the 590 discharges that have it: low 170, middle 135, high 285; share low 0.221 up to 31991, 0.348 in 32002 to 32310, 0.524 from 32312.

| statement | value | verdict | expectation |
|---|---|---|---|
| G1 V1, S: TX over T0, high minus low, above zero | +0.151 [-0.017, +0.277] | does not hold | holds |
| G2 V1, S: TE over T0 in the low class contains zero or lies below | +0.165 [+0.043, +0.302] | does not hold | holds |
| G3 V1, S: TE over T0 in the high class above zero | +0.240 [+0.152, +0.317] | holds | holds |
| G4 V1, D650: high minus low above zero | +0.253 [+0.154, +0.354] | holds | holds |
| G5 V1, S, shots up to 32310: high minus low above zero | +0.141 [-0.046, +0.270] | does not hold | uncertain, leaning to holds |
| G6 V3, S: TX over T0 in the high class above zero | +0.287 [+0.194, +0.396] | holds | holds |
| G7 V1, S, from 32312: TG over T0 above zero | +0.010 [-0.198, +0.112] | does not hold | uncertain |
| G8 V4 (fit to 31991), S: TG over T0 above zero | -0.025 [-0.163, +0.052] | does not hold | uncertain |
| G9 V1, S: TG not worse than TX | -0.067 [-0.128, -0.027] | does not hold | holds |

Reported: TX over T0 by class under V1, S +0.218 / +0.292 / +0.370 and D650 +0.143 / +0.319 / +0.396; inside the last block low +0.084, high +0.154; Spearman of C_E with the per-discharge skill +0.183 [+0.071, +0.283] (444 discharges).

Reading. The coherence goes weakly with the increment, but under the schedule rule the fast diagnostics reduce the MSE in the low class too, the ECEI half included, and switching them off where the coherence is low costs accuracy. The block comparison of `b75_last_block.md` stays a description. By the rules registered beforehand the paper reports the last block under the schedule as a limit without an explanation and proposes no gated estimator. What would decide it: toroidal field and ECEI settings per shot, which are not in the files. Expectation of G1, G2 and G9 not met.

### Rows in which the ECEI channels do not move (unregistered diagnostic, 2026-11-20)

`docs/paper/b75_stuck_ecei.md`. Found on the example discharge of the manuscript (31076: the four ECEI channels are constant until about 2 s). Over all files the ECEI is constant for at least ten rows in 0.035 of the grid rows, in 60 of 641 discharges (6 of them for more than half of their rows); the BES channels never are. The estimators were fitted and scored with these rows in. Stored estimates of B.75 (V1, L5): where the ECEI is constant at the row or at the anchor row the increment is +0.160 against +0.334 (schedule rule) and +0.059 against +0.348 (history 0.65 s old). It does not account for the last block (ECEI constant in 0.017 of its grid rows; +0.039 on the rows where it moves). No interval.

### B.81: the two controls and the two halves under the stale-history rules, every design (2026-11-20)

Registration `PREREGISTRATION_B81.md` with addendum 1 (written before any run). CPU: 220 boosted regressions of the kind of B.75 on B.75's rows, inputs and units, target L5, rules D650 and S; the four checks passed before the readout (`docs/paper/b81_checks.md`). Run by an AI agent under the author's supervision. Output `docs/paper/b81_controls.{md,json}`; the session lead's checks `docs/paper/b81_lead_checks.md` (all seventeen values pooled again from the stored estimates). Intervals with shot clusters resampled.

| statement | value | verdict | expectation |
|---|---|---|---|
| C1 V1, D650: permuted fast inputs, TXp over T0, contains zero or lies below | -0.004 [-0.008, +0.001] | holds | holds |
| C2 V1, S: the same | -0.000 [-0.010, +0.009] | holds | holds |
| C3 V1, D650: TX over TXo (fast inputs of the anchor row) above zero | +0.273 [+0.229, +0.317] | holds | holds |
| C4 V1, S: the same | +0.276 [+0.215, +0.333] | holds | holds |
| C5 V2, D650: the same | +0.298 [+0.232, +0.363] | holds | holds |
| C6 V3, D650: the same | +0.426 [+0.307, +0.562] | holds | uncertain, leaning to holds |
| C7 V4 (fit to 31991), D650: the same | +0.357 [+0.235, +0.523] | holds | uncertain |
| C8 V2, S: the same | +0.296 [+0.205, +0.391] | holds | holds |
| C9 V3, S: the same | +0.274 [+0.192, +0.373] | holds | uncertain, leaning to holds |
| E1 V2, D650: TE over T0 above zero | +0.163 [+0.090, +0.231] | holds | holds |
| E2 V3, D650: TE over T0 above zero | +0.058 [-0.067, +0.166] | does not hold | uncertain, leaning to holds |
| E3 V2, D650: TB over T0 above zero | +0.225 [+0.163, +0.285] | holds | holds |
| E4 V3, D650: TB over T0 above zero | +0.122 [+0.028, +0.234] | holds | uncertain |
| E5 V2, S: TE over T0 above zero | +0.181 [+0.098, +0.257] | holds | holds |
| E6 V3, S: TE over T0 above zero | +0.160 [+0.066, +0.257] | holds | uncertain, leaning to holds |
| E7 V2, S: TB over T0 above zero | +0.221 [+0.157, +0.297] | holds | holds |
| E8 V3, S: TB over T0 above zero | +0.171 [+0.119, +0.229] | holds | uncertain |

Reported: TXo over T0 under D650 is +0.087 [+0.033, +0.134] (V1), -0.023 [-0.103, +0.047] (V2), -0.411 [-0.940, -0.118] (V3), -0.187 [-0.470, +0.048] (V4, fit to 31991); under S +0.070 [+0.006, +0.136] (V1), -0.005 (V2), -0.059 (V3). The permuted control excludes zero in one cell that is not a statement (S, V4 fit to 32310: +0.029 [+0.002, +0.064]).

Reading. With the fast inputs permuted the increment is zero under both head rules. The present samples are needed under every design; fast samples taken when the newest visible CES value was taken add a little under the random design only and do harm in time order. Each half alone fails one design at D650 (ECEI under rolling origin, BES under the chronological design, B.75); only the two together exclude zero under all four. The manuscript's sentence on the ECEI half names the designs in which it holds.

### The manuscript rewritten on the new line (2026-11-20)

`docs/paper/ppcf/main_ppcf.tex` and `supp_ppcf.tex` were rewritten from scratch on sections 1 to 10 of `PAPER_DESIGN.md`: proposal (while the CES value is late or sparse, the fast diagnostics of the present at the same radius carry the level of the pedestal-top ion temperature), demonstration, limits. Title: "Causal nowcasting of the charge-exchange ion temperature on the pedestal-top chord of KSTAR with co-located fast diagnostics". Sections: introduction with the two lines of earlier work; data and the reported value; method (target L5, availability rules, estimators, two controls, four designs, cluster intervals, registration); results 4.1 to 4.6 (cost of a stale history, which diagnostic and beyond the beam waveforms, other divisions and time order, MAST, sparse CES, where it does not hold); discussion; conclusion. Eight figures, nine tables in the main text.

How it is built. No number is typed: `docs/paper/ppcf/make_tables_v2.py` reads the result files of B.73 to B.81 and writes every table and number macro to `gen_v2.tex` with the provenance of each printed token in `gen_v2_tokens.json`; `fig_v2.py` draws the figures (`figv2_*`, each with a json of the plotted numbers); `check_numbers_v2.py` checks every number typed in the tex files against those and the result md files. The example discharge of the figures is chosen by a rule (among the test discharges of one re-partition with at least 500 scored rows under the schedule rule and three notches, the one with the median increment: shot 31076). The data section of the supplement is cut by `make_supp_fragment.py` from the supplement of the earlier line, which is kept frozen in `docs/paper/ppcf/earlier_line/`.

Two things were added while writing: the unregistered diagnostic on rows with a constant ECEI and the registered batch B.81 (entries above). One basis sentence of the discussion is new and is a statement of expectation, not a result: by the electron collision time of Wesson's Tokamaks multiplied by half the ion-to-electron mass ratio, electrons and deuterons exchange energy within ten to a few tens of milliseconds at a density of a few 10^19 m^-3 and an electron temperature of several hundred eV, which is short against the age of the CES value.

Not carried from the earlier supplement: the rotation, the measurement regime, the design of the recurrent network and the replay with stopped updates; their numbers belong to the earlier populations and target.

### Independent grading, round 16, and B.82 (2026-11-20)

**Grading.** A separate agent in the role of a PPCF referee read the two frozen PDFs of the rewritten manuscript (commit ab344d7) and nothing else: PPCF accept 15 %, Fusion Engineering and Design 32 %, Review of Scientific Instruments 27 %, sent to referees 60 %; major revision. The report, its ranked concerns and the eleven places it found stronger than their table are in `docs/paper/review_loop/grade16_20261120.md`; the sentences were corrected (commit 7737a7b).

**B.82.** Registration `PREREGISTRATION_B82.md` with addendum 1 (23 statements; the commit message of the registration says thirty, which is wrong). CPU: stored estimates of B.75, B.79 and B.81 for parts A and B, 520 boosted regressions for parts C and D; the five checks passed before the readout (`docs/paper/b82_checks.md`). Run by an AI agent under the author's supervision. Output `docs/paper/b82_localize.{md,json}`; the session lead's checks `docs/paper/b82_lead_checks.md`. Target L5; intervals with shot clusters resampled.

| statement | value | verdict | expectation |
|---|---|---|---|
| A1 V1, D650, rows more than 300 ms after a notch: TX over T0 above zero | +0.264 [+0.215, +0.314] | holds | holds |
| A2 V1, S, the same rows | +0.334 [+0.264, +0.404] | holds | holds |
| A3 V3, D650, the same rows | +0.077 [-0.109, +0.200] | does not hold | uncertain, leaning to holds |
| A4 V1, D650, rows with no notch before | +0.417 [+0.314, +0.504] | holds | holds |
| A5 V1, S: increment up to 100 ms after a notch minus that above 300 ms, above zero | -0.110 [-0.214, -0.020] | does not hold | uncertain, leaning to holds |
| A6 V1, D650, lowest third of the change since the anchor row | +0.191 [+0.125, +0.255] | holds | uncertain |
| A7 V1, D650: highest third minus lowest third above zero | +0.221 [+0.148, +0.298] | holds | holds |
| A8 V1, S, rows more than 300 ms after a notch: TXN over TN above zero | +0.256 [+0.184, +0.323] | holds | holds |
| B1 V1, D650: MAE(TX) / MAE(T0) below one | 0.826 [0.795, 0.856] | holds | holds |
| B2 V1, D650: ratio of the median absolute errors below one | 0.849 [0.802, 0.892] | holds | uncertain |
| B3 V3, D650: MAE ratio below one | 0.951 [0.876, 1.044] | does not hold | uncertain, leaning to holds |
| B4 V1, S: MAE ratio below one | 0.828 [0.791, 0.860] | holds | holds |
| B5 V1, D650: share within 10 % of the level, TX minus T0, above zero | +0.049 [+0.036, +0.065] | holds | uncertain, leaning to holds |
| B6 V1, D650: share of discharges with a positive skill of their own above one half | 0.831 [0.785, 0.877] | holds | holds |
| B7 V4 (fit to 31991), D650: MAE ratio below one | 0.867 [0.719, 0.998] | holds | uncertain |
| B8 V1, D650: ratio of the 90th percentiles below one | 0.815 [0.779, 0.850] | holds | holds |
| C1 V1, D650: TM (Mirnov scalars in place of the fast ones) over T0 contains zero or lies below | +0.084 [+0.046, +0.119] | does not hold | uncertain |
| C2 V1, D650: TX over TM above zero | +0.275 [+0.232, +0.316] | holds | holds |
| C3 V3, D650: TX over TM above zero | +0.126 [-0.018, +0.213] | does not hold | uncertain, leaning to holds |
| C4 V1, D650: TXs (fast inputs of another discharge at the same time) over T0 contains zero or lies below | -0.083 [-0.129, -0.046] | holds | holds |
| C5 V1, S: the same | +0.019 [-0.012, +0.054] | holds | holds |
| C6 V1, D650: TXM over TX contains zero or lies below | +0.022 [+0.004, +0.039] | does not hold | uncertain, leaning to holds |
| C7 V1, S: TX over TM above zero | +0.273 [+0.211, +0.329] | holds | holds |

Reported (V1, D650): absolute error of T0 / TX / the frame's own reading: MAE 142.3 / 117.6 / 91.9 eV, median 84.9 / 72.0 / 43.4 eV, 90th percentile 336.1 / 274.0 / 217.8 eV; shares within 10 % of the level 0.390 / 0.439 / 0.593, within 20 % 0.624 / 0.677 / 0.785. Increment by time since the last notch (up to 100 ms / 100 to 300 ms / above 300 ms / no notch before): V1, D650 +0.452 / +0.398 / +0.264 / +0.417; V1, S +0.224 / +0.342 / +0.334; V3, D650 +0.293 / +0.257 / +0.077 / +0.372. The three BES radial columns alone: +0.238, +0.246, +0.220. Over ten seeds of the regression the chronological increment (fit to 31991) lies between +0.216 and +0.239 at D650 and between -0.074 and +0.041 under S.

Reading. Under the random design the increment is not confined to the rows after a notch, is present in the median error and in 0.831 of the discharges, and survives the beam scalars on the rows far from a notch. Under rolling origin two of these readings are not established (A3, B3). The estimate is less accurate than the reading of the frame itself in every robust measure (the equality in RMSE came from the large excursions of the reading). The Mirnov signals, which are not at the radius, carry an increment a quarter the size of that of the fast diagnostics: by the rule registered beforehand the manuscript says so and no sentence claims that the increment is specific to the radius. Fast inputs of another discharge at the same time give none.

### Independent grading, round 17 (2026-11-20)

A fresh grader with the instructions of round 16 read the two PDFs of the manuscript with B.82 written in (commit 1ba9a8b): PPCF accept 13 %, Fusion Engineering and Design 30 %, Review of Scientific Instruments 22 %, sent to referees 55 %; reject and resubmit. Round 16 before B.82: 15 %, 32 %, 27 %, 60 %. The report, its ranked concerns and the places it found inconsistent are in `docs/paper/review_loop/grade17_20261120.md`; the sentences and captions were corrected. Its three highest-ranked changes are a validation of the target with the schedule rule in time order as the primary analysis, a shorter paper on what a present-time signal is worth as a function of the age of the CES value without "co-located" in the title, and new data. None was started; they wait for the author's decision.

### The author's framing restored, B.83 and B.84 (2026-11-20 night to 2026-11-21)

**Framing.** The author (2026-11-20 night): the framing set at the start is the better one. `PAPER_DESIGN.md` sections 1, 4 and 9 were rewritten for it (commit e2019b2): nowcasting at the pedestal top, the best model against interpolation and the like, at 10 ms in real time, with a demonstration of use where it applies. The line of the afternoon (bridging a late or sparse CES value) is dropped; its results move into the section on use.

**B.83** (`PREREGISTRATION_B83.md`, addendum 1; CPU; stored estimates of B.73 and the interpolation references of B.70's code path; checks passed; `docs/paper/b83_interp.{md,json}`; lead's checks `docs/paper/b83_lead_checks.md`). Target: the value of the frame, previous frame visible. Intervals with shot clusters resampled.

| statement | value | verdict | expectation |
|---|---|---|---|
| I1 V1: TX over PCHIP above zero | +0.278 [+0.235, +0.315] | holds | holds |
| I2 V1: TX over the offline GP above zero | +0.059 [+0.026, +0.096] | holds | holds |
| I3 V2: TX over PCHIP | +0.261 [+0.217, +0.297] | holds | holds |
| I4 V2: TX over the offline GP | +0.037 [+0.006, +0.070] | holds | uncertain, leaning to holds |
| I5 V3: TX over PCHIP | +0.224 [+0.165, +0.268] | holds | holds |
| I6 V3: TX over the offline GP | -0.007 [-0.046, +0.028] | does not hold | uncertain |
| I7 V4 (fit to 31991): TX over PCHIP | +0.153 [+0.065, +0.228] | holds | holds |
| I8 V4 (fit to 31991): TX over the offline GP | -0.035 [-0.082, +0.014] | does not hold | uncertain |
| I9 V1: T0 over PCHIP above zero | +0.209 [+0.162, +0.243] | holds | holds |
| I10 V1: T0 over the offline GP contains zero or lies above | -0.031 [-0.056, -0.013] | does not hold | holds |
| I11 V1: median absolute error, TX / offline GP, below one | 1.059 [1.014, 1.105] | does not hold | does not hold |
| I12 V1: median absolute error, TX / PCHIP, below one | 1.043 [0.974, 1.104] | does not hold | does not hold |
| I13 V1: MAE, TX / offline GP, below one | 0.977 [0.955, 1.001] | does not hold | uncertain |
| I14 V1: 90th percentile, TX / offline GP, below one | 0.947 [0.925, 0.970] | holds | holds |

Reported, V1 (hold, PCHIP, causal GP, offline GP, T0, TX): RMSE 231.8, 204.9, 191.9, 179.4, 182.2, 174.1 eV; median absolute error 45.3, 38.1, 44.0, 37.5, 41.7, 39.7 eV; share within 10 % of the value 0.545, 0.592, 0.571, 0.618, 0.591, 0.609.

Reading. The nowcast, which does not wait for the next value, has a lower MSE than PCHIP under all four designs and than the offline GP under the random and the cluster-blocked design; in time order it is level with the offline GP. The advantage is in the mean squared error and the upper tail; the median error of the interpolation that waits is slightly lower. The history-only estimate does not reach the offline GP; the fast diagnostics are what carries the nowcast past it.

**B.84** (`PREREGISTRATION_B84.md`, addendum 1; CPU, 60 fits; checks passed; `docs/paper/b84_onefit.{md,json}`; lead's checks `docs/paper/b84_lead_checks.md`). The author's remark: the second stage is bound to rules. Target L5; V1 = re-partition r2.

| statement | value | verdict | expectation |
|---|---|---|---|
| U1 V1, D650: one fit across ages not worse than the per-rule TX by more than 5 % | +0.000 [-0.021, +0.022] | holds | uncertain, leaning to holds |
| U2 V1, D10: the same | -0.062 [-0.087, -0.039] | does not hold | uncertain, leaning to holds |
| U3 V1, S: the same | +0.047 [+0.008, +0.085] | holds | uncertain |
| U4 V1, D650: one fit over the fit made with the previous frame visible and applied | +0.502 [+0.403, +0.568] | holds | holds |
| U5 V1, D650: TXall over T0all | +0.359 [+0.304, +0.403] | holds | holds |
| U6 V1, K10 (not in the mixture): TXall over the per-rule T0 | -0.036 [-0.138, +0.067] | does not hold | uncertain, leaning to holds |
| U7 V3, D650: one fit not worse than the per-rule TX by more than 5 % | +0.080 [+0.050, +0.114] | holds | uncertain |
| U8 V1, D650: raw windows not worse than the hand-made scalars by more than 5 % | -0.327 [-0.432, -0.230] | does not hold | uncertain |
| U9 V3, D650: without absolute levels over the per-rule TX | +0.089 [+0.033, +0.154] | holds | uncertain |
| U10 V4 (fit to 31991), S: without absolute levels over T0 | -0.004 [-0.156, +0.095] | does not hold | uncertain, leaning to does not hold |
| U11 V1, D650: without absolute levels not worse than the per-rule TX by more than 5 % | -0.032 [-0.059, -0.001] | does not hold | uncertain |

Reading. One fit across the ages of the history serves the conditions it was fitted on (equal at 0.65 s, lower under the schedule and in time order, 6 % higher with the previous frame visible) and does not carry to a scenario it was not fitted on (a value every tenth frame: 163.5 against 144.0 eV). A tree on raw windows is far worse than on the hand-made scalars. Without the absolute levels the estimate is better in time order and slightly worse in random folds; the last block under the schedule is not recovered. By the registered rule the paper's estimator is not changed. The author's remark is borne out: the second stage has to be refitted for the condition of use; a model with state trained across scenarios is the open step and needs a GPU.

## 18th independent grading; handover documents (2026-11-21)

**Handover.** On the author's word ("인수인계를 위해서 readme 같은거랑 관련해서 필요했던 것들 정리해줘 나중에는 codex에서도 이거 실험 진행할수 있게"; "GPU는 그러면 나중에 인수인계 문서에만 적고 추후에 돌리자"): `HANDOVER.md`, `AGENTS.md`, `docs/paper/review_loop/GRADER_PROMPT.md` (commit d3fcb2b). No GPU run was started; the recurrent network trained across availability conditions (to be registered as B.85) is specified in `HANDOVER.md` section 9.1. B.86 (MAST from the core to the edge, registered in bc0b95e) was handed to an implementing agent.

**Grading.** `docs/paper/review_loop/grade18_20261121.md`: the manuscript on the author's framing (commit ab0bf1a, main text 17 pages, supplement 28), the fixed prompt. Major revision; PPCF accept 15 %, FED 35 %, RSI 25 %, sent to referees 60 % (rounds 16 and 17: 15 %, 32 %, 27 %, 60 % and 13 %, 30 %, 22 %, 55 %).

**Corrections in the manuscript** (no number of a table changed): the count 627 of the abstract explained in section 4.1 (222,005 of 222,672 rows, 627 of 635 discharges have a reference that waits); "level with" replaced by "no difference is established"; the reading paragraph of the discussion rewritten by design; "the lowest error found" scoped to the previous frame visible; the delay of about a second attributed; the recurrent network alone separated from its two-stage form; the chronological lower end of the increment on the value of the frame named; notches, Mirnov and the sparse rule scoped by design; the rolling-origin fifth below its control named in the main text; MAST Thomson no longer called a fast input; the offline GP specified; the supplement's title and its definition of the level scalar corrected; Bishop 1993, Svensson 1999 and Abbate 2021 cited. Main text 18 pages, supplement 28; `check_numbers_v2.py` passes.

**Verdict.** The estimate for PPCF stays inside the range of rounds 11 to 17. Two analyses that the grader asks for need only the files in hand and belong to the first question of the paper: the same regression with the next values visible (accuracy against the permitted lag), and the main table re-scored without rows after a notch and at low BES level. Neither is registered yet.

## Cleanup of the repository and the memory; B.86 (2026-11-21)

**Cleanup.** On the author's word ("헷갈릴만한 옛연구들은 repo 및 메모리에서 최대한 제거"; "연구 배선도 확인해서 고아 노드가 필요 없다면 제거하고 … readme에 배선 잘된 방향대로 정리해서 작성"): the wiring of the repository was computed from real imports and file reads; what the current line (D1, B.70 onward, the manuscript chain, the presentation builders, the tests of kept code) does not reach was removed in two passes (commits 7a23d95 and 44b74d4; 473 files, 47 experiment directories). The records were cut to the current line and `README.md` was rewritten as the wiring. Everything removed is in the history up to e86adaf.

**B.86, MAST from the core to the edge** (`PREREGISTRATION_B86.md`, addendum 1; `docs/paper/b86_mast_radius.md`, `b86_checks.md`, `b86_lead_checks.md`). The estimator of B.76 refitted in five bands by distance inside the edge of the Thomson density profile (1.5 to 5, 5 to 10, 10 to 20, 20 to 35, 35 to 60 cm; 145,048 to 1,074,935 rows), target the level of the following frames, previous frame visible (D5) and history 100 ms old (D100), designs M1 and M3, sessions resampled.

| statement | result | verdict | expectation |
|---|---|---|---|
| R1, five bands: M1, D100, TX over T0 above zero | +0.296 [+0.246, +0.354] in E to +0.704 [+0.671, +0.735] in B3 | holds in all five | holds |
| R2, five bands: M1, D5 | +0.104 [+0.074, +0.135] in E to +0.198 [+0.182, +0.213] in B4 | holds in all five | holds in E, uncertain inside |
| R3, five bands: M3, D100 | +0.209 [+0.095, +0.305] in E to +0.658 [+0.585, +0.716] in B3 | holds in all five | holds in E, uncertain inside |
| R4, five bands: M1, D5, T0 over the causal GP | +0.180 [+0.119, +0.244] in E to +0.312 in B4 | holds in all five | holds |
| R5: the cost of staleness is larger in E than in B5 | -2.272 [-2.622, -1.927] (E 1.678, B5 3.951) | does not hold | holds |
| R6: the Thomson increment at D100 is larger in E than in B5 | -0.304 [-0.357, -0.245] | does not hold | uncertain |
| R7: the share of TX within 10 % of the level is larger in B5 than in E | +0.133 [+0.116, +0.149] (B5 0.673, E 0.541) | holds | holds |

Reported: against the references that wait, on the value of the frame, the nowcast has the higher MSE on MAST (TX over PCHIP -0.034 [-0.120, +0.047] in E, -0.190 to -0.104 inside with intervals below zero). The four checks passed before the readout; the lead's recomputation agrees (400,657 rows of 266 shots rebuilt from the value arrays, statement values equal, interval ends within 0.006).

**Verdict.** On MAST the same estimator applies from the core to the edge, and Thomson scattering at the chord's radius adds to a stale charge-exchange history at every radius. Against the expectation, the cost of a stale history and the Thomson increment are larger inside than at the edge. The share of estimates within 10 % of the level is higher inside and reaches 0.680 at most. These are statements about MAST; no statement is made about the KSTAR core.

**B.86 in the manuscript.** A paragraph in the section on a second device (the estimator from the core to the edge on MAST; the scope paragraph of the discussion points to it) and two tables in the MAST section of the supplement (per band: radius, level, rows, RMSE, share within 10 %, cost of staleness, the Thomson increment under two rules and two designs, and the estimator against the two references that wait; the three statements across bands). The table sits in the supplement because the main text is at 18 pages. The count of registered statements in the supplement now covers B.73 to B.86: 170 registered, 123 hold, 47 do not; of 99 definite expectations 87 were met. One remaining sentence that stated equivalence with the offline GP in time order now says that no difference is established. Main text 18 pages, supplement 29; check_numbers_v2 passes.
