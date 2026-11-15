# PREREGISTRATION MAST gate — is a hidden 5 ms CXRS frame an independent measurement, and do CES-only causal estimators leave headroom

Registered 2026-11-15, before any value of any gate shot is read. Committed before the analysis
code. Follows `docs/paper/review_loop/venue_plan/PLAN_review.md` section 5, steps 4-6. The request
for further KSTAR data is dropped (not obtainable); the target venue is PPCF and FAIR-MAST is the
second device.

## 1. Data

- Source: FAIR-MAST level 1, source `act` (time-resolved charge exchange), anonymous HTTPS,
  CC BY-SA 4.0. Downloaded by `mast/fetch_mast.py` into `data/mast/act/{shot}.npz`.
- Shot list: `mast/gate_shots.txt` (300 shots, 75 per campaign M6-M9, seed 20261116, catalogue
  quality "Not Checked", the 13 pilot shots excluded), fixed in commit 7fd034c from the catalogue
  alone. 264 carry the `act` arrays, 36 do not (store present, arrays absent).
- State of knowledge at registration: values were read for the 13 pilot shots only
  (`PLAN_review.md` section 1). For the gate shots only file existence and byte counts are known.
  The other 7,657 `act` shots are being downloaded and are not read before the gate report.
- Tune / test: inside each campaign the gate shots that carry data are sorted by shot number; even
  positions (0, 2, ...) are TUNE, odd positions are TEST. Hyperparameters are chosen on TUNE only;
  every reported number is on TEST.

## 2. Definitions (fixed)

- Units: T_i in eV, V_i in km/s (stored m/s divided by 1000).
- Good sample (T_i): finite, T_i > 0, reported error > 0, T_i < 5000 eV, relative error < 0.1
  (population G, primary) or < 0.5 (population W, co-reported).
- Good sample (V_i): the T_i sample is good in the same population, V_i finite, reported V_i
  error > 0.
- Usable shot: median frame spacing within 5.0 +- 0.1 ms. Beam block: first to last frame with at
  least 10 good chords; at least 20 frames long. Scored chord: good in at least 90 % of the block
  frames. A shot with no block or no scored chord contributes nothing and is counted.
- Decimation: block frames at even positions are visible, odd positions are hidden. An estimator
  for the hidden frame at time t on a chord uses visible samples of that chord only; causal arms
  use times < t, offline arms use both sides.
- Scored row: hidden sample good, visible samples at t - 5 ms and t - 15 ms good, visible sample at
  t + 5 ms good. One common row set for every arm, per target and population.
- `ss_chisq` is not used in any rule: its column count (160) differs from the chord count (96) in
  the one shot whose shapes are known and its scale has not been examined. It is reported
  descriptively only if its columns can be mapped to chords.

## 3. Gate (a): independence of the hidden frame

Reported errors are not used here. For spacing k frames (k = 1, 2, 4) and each scored chord, the
second difference d_k(t) = y(t + k) - 2 y(t) + y(t - k) over triples of good samples in the block;
s2_k = (1.4826 x median absolute deviation of d_k)^2 / 6 per chord with at least 12 triples; per
shot S_k = median of s2_k over chords.

If the fit noise is white in time with variance s2 and the second-difference variance of the plasma
signal grows as c k^p with p >= 1, c >= 0, then S_1 = s2 + c and
S_2 - (S_4 - S_2) / 2 = s2 + c (2^p - 2^(p-1) (2^p - 1)) <= s2 + c, so

    T_a = (S_1 - S_2 + (S_4 - S_2) / 2) / S_2  >=  0.

Positive noise correlation rho_1 between adjacent 5 ms frames lowers T_a by about (4/3) rho_1
(noise-dominated limit). Statistic: median of T_a over TEST shots, 95 % interval from 2,000
bootstrap resamples of shots (seed 20261116), for T_i and V_i, population G.

- **Gate (a) fails** for a target if the interval lies entirely below 0. Then a hidden frame shares
  noise with its neighbours; decimation scores on MAST are not quoted as "a measured answer at the
  filled instant", and MAST is used for the regime and coupling study only.
- Otherwise gate (a) passes. The test is one-sided and conservative (signal variation raises T_a),
  so passing is "no adjacent-frame noise correlation detected at this power", stated with the
  interval. The implied bound rho_1 <= -(3/4) x (interval lower bound) is reported when negative.
- Co-reported, not gating: the median over rows of d_1^2 divided by its reported-error variance
  (4 sigma_t^2 + sigma_(t-1)^2 + sigma_(t+1)^2) relative to the Gaussian value 0.455. Pilot value
  0.31 / 0.455 = 0.68. If gate (a) passes and the interval of this ratio is below 1, the reading is
  "reported errors are larger than the frame-to-frame scatter at the median row", and the floor of
  section 4 is quoted both raw and with errors rescaled by the square root of that ratio.

## 4. Gate (b): headroom left by CES-only causal estimators

Arms, all linear in the visible values (weights depend on times and reported errors, not on values):

| arm | definition | tuned on TUNE |
|---|---|---|
| persistence | last visible sample (t - 5 ms) | - |
| extrapolation | line through t - 5 ms and t - 15 ms | - |
| running mean | exponentially weighted mean of visible samples, weight exp(-age / tau) | tau in {5, 10, 15, 25, 40, 70} ms |
| slope-state filter | Kalman filter, state (level, slope), random-walk noise on both, measurement variance = reported error squared | q_level, q_slope on a 6 x 6 log grid |
| causal GP | Matern-3/2 plus constant kernel, measurement variance = reported error squared, last 8 visible samples | length scale in {10, 20, 40, 80, 160} ms, amplitude in {0.5, 1, 2, 4} x target scale |
| offline interpolation | mean of t - 5 ms and t + 5 ms | - |
| offline GP | same kernel, 4 visible samples on each side | same grid |

Tuning criterion: pooled mean squared error on TUNE scored rows, population G, per target.
Target scale for the GP amplitude: median reported error of the TUNE rows.

Readouts on TEST (per target, populations G and W; intervals: 2,000 shot-clustered bootstrap
resamples, seed 20261116):

- MSE skill of each arm against persistence, 1 - SSE_arm / SSE_persistence.
- Reduced chi-square of each arm: squared residual divided by the variance the reported errors
  predict for that arm, sigma_t^2 + sum_i w_i^2 sigma_i^2 (w_i the weights of the arm). Mean and
  median.
- Best causal arm = the causal arm with the lowest TEST MSE in population G.

- **Gate (b) passes** for a target if the 95 % interval of the mean reduced chi-square of the best
  causal arm lies above 1.5 in population G. Then residual structure above the fit noise remains
  and a learned arm with other MAST signals (Mirnov, D-alpha) is registered separately.
- **Gate (b) fails** otherwise: the CES-only filter is at the floor the reported errors set, no
  neural arm is trained on MAST, and the MAST result is the regime contrast with KSTAR.

## 5. Regime number and audit readouts (reported with the gate, not gating)

- Regime number per TEST shot and target: median over scored chords of the lag-1 autocorrelation
  of 10 ms increments of the visible series (KSTAR T_i: -0.41; white noise on a flat signal: -0.5).
- Registered statement R1: on MAST the skill of the slope-state filter against persistence has an
  interval above 0 and the skill of the running mean has an interval below 0 or including 0, for
  T_i and for V_i (population G). KSTAR is the opposite regime (a running mean takes part of the
  PCHIP margin). Both directions are reportable.
- Registered statement R2: across TEST shots the Spearman correlation between the regime number and
  (slope-filter skill - running-mean skill) is positive, interval above 0.
- Audit: ratio of mean to median reduced chi-square for the best causal arm (Gaussian: 2.2; pilot,
  offline interpolation: 7.5); share of the SSE of the best causal arm carried by the 0.1 % of rows
  with the largest squared residual; skill recomputed with those rows dropped from every arm.
- Radius: every skill and reduced chi-square by tertile of major radius among the scored chords.

## 6. Numerical expectations (from the 13 pilot shots; stated so they can fail)

- Usable shots: 85-95 % of the 264; median block 45-60 frames.
- Gate (a): T_a interval includes 0 for T_i (pass); co-reported ratio below 1.
- T_i, population G: extrapolation vs persistence about -0.4; running mean (best tau = 5 ms, i.e.
  persistence) no better than 0; slope-state filter +0.05 to +0.25; offline interpolation +0.35 to
  +0.60; mean reduced chi-square of the best causal arm 2 to 5 (gate (b) passes), median below 1.
- V_i, population G: slope-state filter +0.2 to +0.5; offline interpolation +0.55 to +0.75; mean
  reduced chi-square above 10 (gate (b) passes).
- Mean-to-median ratio above 4 for both targets.

## 7. Disclosures

- Gaps on MAST are made by decimation; KSTAR gaps are natural. This is a contrast of regimes, not
  a replication, and no MAST number is written as support for the KSTAR fast-diagnostic claim.
- Reported errors are taken as stored; section 3 is the only check on them here.
- Catalogue quality "Bad" shots are excluded from the gate by construction of the list.
- No neural network is trained in this registration. CPU only.
- Output: `docs/paper/mast_gate.json` / `.md`, record in `THESIS_RESULTS.md` section 8.
