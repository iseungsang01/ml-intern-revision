# B.73 — what the fast channels add: by target, by validation design, without a network

Registered 2026-11-20, before any run of this batch. CPU only: boosted regressions, small filters and re-reads of stored estimates; no network is trained or run. The session lead designed it. An AI agent implements and runs it under the author's supervision. No existing file is edited.

## 1. Why

The author asked for harsh criticism by reviewers in the role of researchers, followed down to its roots. Three critiques were written on 2026-11-20 (`docs/paper/review_loop/critique_20261120_{referee,pedestal,stats}.md`), each with unregistered read-only checks. Their chains end in four roots that the existing data can address:

| Root | What the reviewers measured (unregistered) |
|---|---|
| The scored quantity is the reported value of the frame, not the level of the temperature | Scored against the following frames, about a quarter of the fast-channel increment remains; with the correction about halved it is about +0.06. The session lead reproduced the first number: retention of the MSE reduction 0.24 [0.12, 0.37] |
| Discharges are treated as exchangeable | The increment sits in shot clusters before shot 32000; on the chronological split the network's increment over its twin is +0.005 [-0.019, +0.028]; boosted trees keep +0.02 to +0.04 |
| The recurrent network was never tested against a learner of equal flexibility | A causal GP with boosted trees on 18 scalars has the MSE of the two-stage estimator |
| What the fast channels co-vary with is not attributed | The reported value follows the beam-emission light of its own exposure; ECEI, which uses neither the beam nor the spectrometer, follows too |

This batch measures all four in one registered table, with no row called primary.

**Status of this registration.** No untouched data exist. The reviewers have already read forms of several numbers below on the same discharges. The batch is a systematic, registered repetition, not a confirmation on fresh data.

## 2. Data

- Rows: the D1 rows of B.70 (`PREREGISTRATION_B70.md` section 2): 639 discharges, T_i, values above 3 keV and held repeats withheld.
- Per-row quantities: the causal GP and hold estimates and the age of the last value (`data/.b64/base.npz`, as B.64 built them); the grid (`seq_data.load_grid_files`, cut, held removed).
- Stored network-based estimates of r2–r4: `data/.b70/stage2_r{r}_f{k}.npz` (network, twin, two-stage estimator, two-stage twin).

**Shot clusters.** A run of delivered discharges whose consecutive shot numbers differ by at most 3. The files carry no date; a cluster stands in for a session. The number of clusters and their sizes are reported.

## 3. Targets

For a scored row t (a D1 row):

| Target | Definition |
|---|---|
| F (frame) | the reported value of row t, as in every earlier batch |
| N1 | the reported value of row t+1, 10 ms later |
| L3 | the mean of the reported values of rows t+1 to t+3, all three measured |
| **L5 (following level)** | the median of the measured values among rows t+1 to t+5, at least three measured |
| L10 | the median of the measured values among rows t+1 to t+10, at least five measured |

Rows t+j are grid rows of the same block, 10 j ms after t. A row without the target is left out of that target's readouts; the counts are reported. No estimator reads any value of row t or later. The later values are independent readings of a quantity that changes slowly; they carry their own scatter, which is independent of every estimate.

## 4. Estimators

**Without a network** (all causal: inputs of row t use CES values before t and fast samples up to t).

| Name | Definition |
|---|---|
| H | hold: the last measured value |
| M5 | the median of the last five measured values |
| G | the causal GP of the earlier batches (not fitted to a target) |
| T0 | G plus a boosted regression of (target − G) on ten history scalars: time step, age of the last value, hold, G − hold, \|G − hold\|, and of the last ten measured values the mean, the standard deviation, hold minus the mean, the last increment, the number present |
| TX | T0's inputs plus the eight fast scalars of B.64 (`run_b33.window_feats`): for the BES array mean and for the ECEI array mean, the level, the departure from the mean of the ten preceding rows, their standard deviation, the one-row increment |
| TB | T0's inputs plus the four BES scalars |
| TE | T0's inputs plus the four ECEI scalars (no beam, no spectrometer) |
| TXlag | T0's inputs plus the eight fast scalars of row t−1 |
| K0, KX | the filters of B.56 (three and five parameters), fitted by minimising the eV mean squared error of the output against the target |

- Boosted regression: `HistGradientBoostingRegressor(random_state=20261116)`, default settings (`run_b37.gbr`). No tuning.
- Each fitted estimator is fitted separately for each target it is scored on. "T0[L5]" is T0 fitted to L5.
- Rows without a complete input set keep G in every tree estimator; their share is reported.

**With the stored networks** (random design only, since the networks exist only there).

- As built: network, twin, two-stage estimator, two-stage twin, scored against every target.
- With the gain fitted: twin + λ (network − twin), and two-stage twin + λ (two-stage estimator − two-stage twin). λ is one number per fold and target, the least-squares slope of (target − history-only estimate) on the correction, taken on the rows of the re-partition's other four folds. Limit: the networks of those folds had this fold's discharges in training; for one scalar this is accepted and stated.

## 5. Validation designs

| Design | Fit and test |
|---|---|
| V1 random | the discharge-level five folds of r2, r3, r4 (the paper's) |
| V2 cluster-blocked | whole shot clusters assigned to five folds of near-equal discharge counts, three random assignments (seeds 1, 2, 3) |
| V3 rolling origin | the campaign cut into fifths by shot order with equal discharge counts; fifths 2 to 5 each scored by a fit on the earlier fifths only |
| V4 chronological | the split of B.5 and B.69: fit on discharges up to shot 31991, score from shot 32312; beside it, fit on discharges up to shot 32310 |

- T0, TX, K0, KX: every design, targets F and L5; under V1 also N1, L3, L10 (trees only).
- TB, TE, TXlag: V1 and V4, targets F and L5.
- For a tree estimator, "fit" means the boosted regression is fitted on the fit discharges' rows; G needs no fit. For the filters, the parameters are fitted on the fit discharges.

## 6. Readouts

One table: rows = design, columns = target × learner, cells = the fast-channel increment (the estimator with fast inputs over the same estimator without them, MSE skill and MSE reduction in %, on identical rows).

Also:
- skill over G and over H, RMSE and median absolute error, per estimator, design and target;
- under V1, every cell by chronological block (up to shot 31991; 32002–32310; from 32312) and by fifth of the BES level of the row (a predictor, not the scored value);
- the fitted λ, and the filter gains b_c and b_g fitted to F and to L5;
- under V1 and target F, the reproduction of the D1 values of B.70 for the stored estimates (check).

**Intervals.** 95 %, 20,000 draws, two forms beside each other:
- discharges resampled (as the paper does);
- shot clusters resampled.

Under V1 a re-partition is drawn first, under V2 an assignment; under V3 and V4 the test discharges or clusters are resampled. A statement is judged on the cluster form.

**No stratum is defined by the scored value or by later values.** Readouts by "kind of frame" (spike, step, quiet) are not part of this batch.

**Usability of the frame (classification).** Label: grid row t has no usable T_i (missing, or above 3 keV), among grid rows with at least one measured value in the ten rows before. Two boosted classifiers (`HistGradientBoostingClassifier`, default settings): history inputs (time step, age of the last value, the number of measured values among the last ten rows, whether row t−1 had a value), and those plus the eight fast scalars. Area under the ROC curve under V1 and V4; difference with a cluster interval.

## 7. Statements

All on the cluster interval. "Increment" = TX over T0 unless said otherwise.

| Id | Statement | Expectation |
|---|---|---|
| Q1 | V1, as built: the increment of the two-stage estimator over its twin against L5 is less than half of its increment against F on the same rows (interval of the ratio of the two MSE reductions below 0.5) | holds (0.1 to 0.4) |
| Q2 | V1, target L5, estimators fitted to L5: the increment lies above zero | holds (+0.03 to +0.07) |
| Q3 | V2, target L5: the increment lies above zero | holds (+0.02 to +0.06) |
| Q4 | V3, target L5: the increment lies above zero | uncertain (+0.01 to +0.04) |
| Q5 | V4, target L5: the increment lies above zero | uncertain (0.00 to +0.03) |
| Q6 | V1, target L5: TE over T0 lies above zero (an increment that needs neither the beam nor the spectrometer) | uncertain (0.00 to +0.03) |
| Q7 | V1, target L5: TX[L5] is not worse than the gain-fitted two-stage estimator (the interval of its skill over it is not below zero) | holds |
| Q8 | V4, target L5: T0[L5] over G lies above zero (what the history estimator adds transfers in time) | holds |
| Q9 | V1 and V4, target F: under each design the increment against F exceeds the increment against L5 (interval of the difference above zero) | holds under V1; uncertain under V4 |
| Q10 | V4: the area under the ROC curve with the fast scalars minus that without lies above zero | holds (+0.01 to +0.05) |

Each statement is judged separately; the table of section 6 is reported whole, whatever the verdicts.

## 8. What goes into the paper (decided now)

- Two targets are named in the text: the reading of the frame (F) and the following level (L5). A sentence about information on the ion-temperature level cites L5. A sentence about predicting the reported value cites F.
- The table of designs is printed whole. The abstract cites, for each claim, the least favourable design under which the claim still holds, and names the design.
- If Q7 holds, the estimator of the paper is the one without a network; the recurrent network becomes a comparison row.
- If Q2 fails: the paper states that the fast channels are not shown to improve the estimate of the level, and what remains is the co-variation with the reading of the frame and the usability flag.
- If Q5 fails while Q2 to Q4 hold: the increment is stated as not established forward in time on this campaign.
- If Q6 holds, the text states which part of the increment a beam-independent diagnostic reproduces; if not, that every part of it needs the beam emission.
- Sentences now in the record that this batch replaces: "most of the network's correction is still present in the next frame" (true of the correction's slope, not of the gain), and the reading of B.70 D1-6 that the first stage's gain is in rows unlike their neighbours (a stratum defined by the scored value).

## 9. Limits known in advance

- Exploratory on used data (section 1).
- The following level is built from reported values. A bias of the analysis that lasts 50 ms or more is in the target too; only an independent measurement removes it.
- Shot clusters are a proxy for sessions.
- V3 confounds "earlier only" with "fewer fit discharges" for the early fifths. V4 is one block of about 95 discharges.
- The network-based rows exist under V1 only. Whether a network trained on L5 does better than the gain-fitted one needs a GPU and is not part of this batch.
- Whether the part that follows the fast channels is the fit, a displacement of the edge profile or the ion temperature is not decided by any readout here. The data that would decide it (fit amplitude and error, neighbouring chords, equilibrium) are not in the files.

## 10. Checks and files

- V1, target F: the stored estimates reproduce the B.70 values within 0.001; T0 and TX reproduce the reviewer's network-free values of `critique_20261120_stats.md` (appendix A, C7) within 0.005.
- No estimator input of row t contains a value of row t or later (tested by changing those values and re-running one fold).
- The folds of V2 keep every cluster whole; the fits of V3 and V4 contain no discharge later than the first test discharge.
- Code: `ces_prediction/experiments/b73_targets/`. Caches: `data/.b73/`. Output: `docs/paper/b73_targets.{md,json}`, checks `docs/paper/b73_checks.md`.
- Seed 20261173, one stream per quantity (crc32 of its name).
