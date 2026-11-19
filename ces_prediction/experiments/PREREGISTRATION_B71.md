# B.71 — when the CES updates stop: free run, own-estimate feedback, and return

Registered 2026-11-19, before any run of this batch. CPU only: stored networks are run forward on rebuilt inputs; nothing is trained. The session lead designed the batch. An AI agent implements and runs it under the author's supervision. No existing file is edited.

Runs after B.70 phase 1, which defines the rows (D1) and re-scores the networks used here. The filter rows wait for B.70 phase 2 (section 2).

## 1. Why

The author asked how the estimator behaves when its estimates are all predictions, compared with when measured values replace them as they arrive. Does the error accumulate and diverge, or does it come back?

Every number reported so far is the update condition: at row t the estimator holds every CES value measured before t. Two replays cover short losses:

- one CES value every 50 ms (B.30, B.48 H1–H2);
- values arriving L ≤ 4 frames late (B.42, B.48 T1–T2).

Nothing covers losses of 100 ms to seconds, the estimator's own estimate fed back as if measured, or the return after updates resume.

## 2. Data and estimators

- **Rows and splits.** D1 (`PREREGISTRATION_B70.md` section 2): 639 discharges, pooled r2–r4 (15 folds), T_i above 3 keV and held repeats withheld, test discharges of each fold.
- **Network.** `data/.b39p_v2_cut_r{r}_f{k}` (seq_v2, train-file standardisation), frozen, CPU.
- **Twin.** `data/.b39p_v2h_cut_r{r}_f{k}` (v2h, no fast channels, per-discharge input standardisation as stored), frozen, CPU.
- **hold.** The last available T_i value.
- **Causal GP.** `baselines_interpolation.predict_gp_causal` on the 16 last available samples, at the stamp of row t.
- **KF0, KFX.** The B.56 filter (`b56_kfx/kfx_model.run`) with the parameters of B.70 phase 2 (item 2.9). A sample that is not available makes no update (predict step only). If B.70 phase 2 has not delivered the parameters, the filter rows are left out and the readout says so.
- **Not included.** The two-stage estimator. Its second stage reads history inputs whose meaning changes under the scenarios, and refitting it per scenario is outside this batch.

## 3. Scenarios

**Availability.** A CES sample measured at row u is available at row t when u ≤ t − 1 (as now) and the scenario keeps it. The six CES input channels (last available value, log1p staleness, observed flag; T_i and V_rot) are rebuilt from the available samples by the rule of `run_b42.ces_channels` (true staleness), generalised to an availability mask. The fast channels and the time step are unchanged. One rule for both targets: the CES analysis delivers T_i and V_rot from one fit.

**Cut points.** In each block of each test discharge: c_j = block start + j × 1.0 s, j = 1, 2, … A cut is kept when at least three T_i samples are available before it and the block continues at least 0.5 s after it. Each cut is a separate replay; rows before c are those of U.

| Scenario | From the cut on | Horizon |
|---|---|---|
| U (update) | every sample available, as now | the D1 rows |
| F (free run) | samples measured at or after c are not available; the network is told their true age | c + 3 s or the block end |
| R (own feedback) | samples measured at or after c are not available. At row t the CES channels hold the estimator's own estimate of row t − 1 (T_i and V_rot), standardised, with staleness time[t] − time[t − 1] and flag 1. Run row by row with the LSTM state carried | c + 3 s or the block end |
| G (gap, then return) | samples measured in [c, c + g) are not available, g = 0.1, 0.3, 1.0 s; after c + g every sample is available again | c + g + 1 s |

R applies to the network and the twin. hold, the causal GP and the filters have no R form.

**Training range of the staleness input.** On the training rows of each fold (D1), the 50th, 99th, 99.9th percentile and the maximum of the T_i staleness input are reported. A time bin whose lower edge exceeds the 99.9th percentile is marked "outside the training range".

## 4. Readouts

The unit is a (scored row, cut) pair. A scored row is a D1 row with T_i measured. For F and R, e = time[t] − time of the last T_i sample available before c. For G, s = time[t] − (c + g), the time since updates resumed.

- **F and R bins of e:** (0, 10], (10, 50], (50, 100], (100, 300], (300, 1000], (1000, 3000] ms.
- **G bins of s:** (0, 10], (10, 30], (30, 100], (100, 300], (300, 1000] ms.

Per bin and estimator: MSE (10³ eV²), RMSE (eV), median absolute error (eV), mean signed error (eV), number of pairs and discharges. Pairs on identical units:

| Pair | Reads |
|---|---|
| network F vs twin F | what the fast channels add after updates stop |
| network F vs hold F | over carrying the last value |
| network F vs causal GP F | over the history-only extrapolation |
| network F vs KFX F, KF0 F | over the filter's free run |
| network F vs network R | telling the true age vs feeding back its own estimate as measured |
| network U vs network F | the cost of losing updates, on the same rows |
| network G vs network U | the return after updates resume |

Comparisons are reported as MSE reduction in % (100 × skill) and RMSE difference (eV), point and 95 % interval.

**Divergence rule.** In a bin, an R estimator is called divergent when the 95 % interval of its MSE over the MSE of hold F (same units) lies above 2, or when more than 5 % of its estimates fall outside [0, 3,000] eV. The share outside [0, 3,000] eV is reported for every estimator and bin.

**Return rule.** For gap g, the network has returned by bin b when the upper bound of the interval of MSE(G) / MSE(U) on the same units is below 1.10.

## 5. Statements

| Id | Statement | Expectation |
|---|---|---|
| S1 | Network F over twin F: the interval of the MSE reduction lies above 0 in the bins (100, 300] and (300, 1000] ms | holds |
| S2 | Network F over hold F: the interval lies above 0 in every bin up to 1,000 ms | holds |
| S3 | The network's R is divergent in no bin up to 3,000 ms | holds (weak expectation) |
| S4 | Network F over network R: the interval lies above 0 in the bin (300, 1000] ms | holds |
| S5 | For g = 0.1, 0.3 and 1.0 s, the network has returned by the bin (30, 100] ms | holds |

Each statement is judged separately; there is no correction for the five. Reported without a statement: U vs F per bin, the filters, the twin's R, the causal GP's F, the mean signed error per bin, and the training range flags.

## 6. Checks (the run stops if one fails)

1. With every sample available, the rebuilt inputs equal `build_blocks` (np.allclose, rtol = atol = 1e-6) on every test file of every fold.
2. U predictions of the network equal the B.70 D1 re-scoring on the shared rows within 0.5 eV on at least 99.9 % of the rows of each fold.
3. With a cut after the block end, F, R and G equal U exactly.
4. R stepped row by row with full availability equals the batched forward within 1e-4 eV.

## 7. Limits

- The networks were trained on the staleness the data have (beam-off frames, block starts). Losses of seconds lie outside that range; section 3 measures where.
- R is a stress test, not an operating mode: the network is told a value is fresh when it is its own estimate.
- Cuts every 1 s give overlapping horizons; the discharge-level bootstrap carries that correlation.
- T_i only. One initialisation per fold.

## 8. Intervals

Two-level bootstrap of B.70 addendum 1 A4: a re-partition drawn among r2–r4, then its test discharges resampled with replacement; units weighted as in `stage2_b64.cell`; 20,000 draws. MSE ratios and reductions from the same draws. Seed 20261171, one stream per quantity (crc32 of its name).

## 9. Files

- Code: `ces_prediction/experiments/b71_scenarios/`.
- Rows: `data/.b71/` (gitignored).
- Checks: `docs/paper/b71_checks.md`. Readout: `docs/paper/b71_scenarios.{md,json}`.
- Cost: F and G are one batched forward per block and scenario; R runs row by row, batched over cuts. Estimated 1–3 CPU hours. The run stops and reports if the projection exceeds 8 hours.
