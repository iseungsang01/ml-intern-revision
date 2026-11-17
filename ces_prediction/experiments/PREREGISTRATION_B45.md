# B.45 — in which rows each fast-channel group carries the increment

Registered 2026-11-18, before any of the readouts below was computed. CPU only; nothing is trained and no network is run (the row-level squared errors of finished runs are read). Designed by the session lead and executed by a delegated agent from this document.

## 1. Why

The ablation of B.39 (arm A, initialisation = split seed; its 16 runs are complete) gives the answer that the registration of B.39 names as the first caveat: a network that reads the beam-emission channels (BES) and the CES history has the fast-channel increment, a network that reads the ECEI channels and the CES history does not on three of four splits. BES is light of the heating beam, and the CES value needs the beam. The next question of a referee is whether the increment carried by BES is the response to interruptions of the beam, or whether it is present in rows that have no interruption behind them. The row-level errors of the ablation networks answer it without a further run.

## 2. Seen before

B.39 arm A, initialisation = split seed, read on 2026-11-18 (the record of B.39 follows when its further initialisations are complete). Over the history-only twin, splits 42, 1, 7, 123: the ECEI-only variant +0.010 [−0.015, +0.034], +0.001 [−0.015, +0.015], +0.084 [+0.066, +0.101], +0.038 [−0.001, +0.073] (D1 holds on 1 of 4); the BES-only variant +0.074 [+0.039, +0.111], +0.026 [+0.007, +0.049], +0.105 [+0.071, +0.134], +0.063 [+0.036, +0.096] (D2 holds on 4 of 4); the full causal variant over the BES-only variant −0.003, −0.004, +0.021, −0.003 (every interval includes zero), over the ECEI-only variant +0.062, +0.022, +0.043 (zero excluded) and +0.024 [−0.010, +0.069].

Earlier: after an absent frame (a beam blip) the measured T_i is 0.819 and the ECEI level 0.926 of the level before in the first row, with the same recovery time (41.0 and 40.2 ms); in the rows 1 to 3 after an absent frame the causal variant (cap 30) has +0.195 to +0.256 over its twin; inside the rows that pass the four value rules its increment is +0.044 to +0.152 (four of four); a BES channel value more than three standard deviations above its 100 ms mean is a burst sample in 0.564 of the cases. Not seen: the skill of the ECEI-only or the BES-only network inside any class of rows.

## 3. Data

T_i, cut population, the four file-level splits, test rows. Per split the row-level squared errors `comparison_errors_test.npz` (`CES_TI_shot`, `CES_TI_row`, `CES_TI_se_model`, `CES_TI_se_gp_causal`, `CES_TI_se_persistence`) of five runs, all with an epoch cap of 80:

- full causal variant `data/.b39_v2_cut_s{s}_i{s}`, ECEI-only `data/.b39_v2e_cut_s{s}_i{s}`, BES-only `data/.b39_v2b_cut_s{s}_i{s}`, no-Mirnov `data/.b39_v2n_cut_s{s}_i{s}`;
- history-only twin `data/.b29a1_v2h_cut_s{s}`.

Checks before the readout (the run stops if one fails): (a) the five runs of a split are scored on the same rows and their reference errors agree; (b) the skills over the twin recomputed on all rows equal those of `docs/paper/b39_primary.md` to three decimals; (c) the meaning of `CES_TI_row` is established by recomputing the squared error of hold from the 10 ms file of five test discharges per split and matching `CES_TI_se_persistence` on at least 99 % of their rows (the convention found is stated in the output).

## 4. Classes of rows

From the 10 ms file of the discharge (definitions already in use: `b32_two_sources/posthoc_b32.py::channel_departures`, `b41_events/run_b41.py::Ctx.after_low_bes`). Each scored row gets the first class that applies:

1. **after an absent frame**: one of the ten rows after a time step of 20 to 40 ms inside a segment;
2. **low beam emission**: within 30 ms after a row whose BES level (mean of the nine channels) is below half the discharge median;
3. **upward departure**: at least one BES channel of the row is more than three standard deviations above its mean over the ten rows before;
4. **downward departure**: at least one BES channel of the row is more than three standard deviations below it;
5. **after a departure**: a row of class 3 or 4 lies in the three rows before;
6. **quiet**: none of the above.

Rows whose departure is not defined (fewer than ten rows of the segment before) are reported as a seventh group and enter no statement.

## 5. Statements

Intervals: 95 %, 2,000 resamples of the test discharges. A statement holds if it holds on at least three of the four splits.

- **R1.** In the rows outside classes 1 and 2 (no beam interruption behind them) the BES-only variant has skill over the twin with the interval above zero.
- **R2.** In the quiet rows (class 6) the BES-only variant has skill over the twin with the interval above zero.
- **R3.** In the rows of class 1 the ECEI-only variant has skill over the twin with the interval above zero.
- Reported, per class and split: the share of the rows; the skill over the twin of the full, the BES-only, the ECEI-only and the no-Mirnov network; the full network over the BES-only and over the ECEI-only network; the BES-only network over the causal GP; the share of the total squared-error reduction against the twin (sum over rows of twin error squared minus network error squared) that falls in each class, for the BES-only and for the full network, with intervals; the root-mean-square error of hold in each class.

## 6. What goes into the paper

- R1 holds: the increment carried by the beam-emission channels is stated as present in rows with no beam interruption in the 100 ms before, with the numbers; the beam stays the first caveat in the form "BES depends on the beam and on the density". R1 fails: the increment is stated as established after beam interruptions only, and the claim of the paper is bounded to them.
- R2 holds: stated for quiet rows as well. R2 fails: the paper says that the increment sits in rows with a large departure of a fast channel or shortly after one.
- R3 holds: the response to a beam interruption is read from a diagnostic that does not use the beam as well. R3 fails: stated; the ECEI channels as sampled do not carry it for the network.

## 7. Expectations

Shares of rows: class 1 0.03 to 0.08, class 2 below 0.04, classes 3 and 4 0.01 to 0.04 each, class 5 0.03 to 0.10, quiet 0.75 to 0.90. R1: +0.02 to +0.09, holds on three or four splits. R2: 0.00 to +0.05, holds on two or three splits (uncertain). R3: +0.03 to +0.20, holds on two to four splits. Share of the squared-error reduction of the BES-only network in classes 1 and 2 together: 0.25 to 0.60.

## 8. Limits known in advance

One initialisation per network; the spread between initialisations of the causal variant is read in B.39. The classes are defined from BES, which the ECEI-only network does not read. Classes 1 to 4 hold few rows, so their intervals are wide. The thresholds are those of earlier readouts and were not tuned here. The test rows of these splits have been read by D1 and D2 and by earlier registered readouts.
