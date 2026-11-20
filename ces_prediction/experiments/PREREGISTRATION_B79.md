# B.79 — the beam waveforms as inputs: what the fast diagnostics add beyond what the actuators already say

Registered 2026-11-20, before any run of this batch and before any estimate with a beam input has been computed. CPU only, boosted regressions on cached rows; no network. The session lead designed it after the NBI files arrived that day (`docs/paper/nbi_files.md`). An AI agent implements and runs it under the author's supervision. No existing file is edited.

## 1. Why

B.75: with the CES history 0.65 s old, the fast diagnostics of the present (BES, ECEI) reduce the MSE of the level estimate by 19 to 34 % under four designs. A referee's first question to that result: the level of the ion temperature changes over a second because the heating changes (a beam source switches, the diagnostic beam is notched), and the beam waveforms are known in real time without any diagnostic. Is the increment of the fast diagnostics what the beam waveforms already give?

Until 2026-11-20 the files had no beam signal, so the question could not be asked. The NBI files give power, voltage and current of each source on the rows of the shot files (same stamps in all 641 shots). They show that an absent frame is a notch of the two NBI-1 sources that run (B and C), and that the first row after one carries a power sample taken while the beam returns (0.61 of the level).

This batch adds beam scalars to B.75's estimators and reads two things: what the beam waveforms add to the history alone, and what the fast diagnostics add on top of the history and the beam waveforms. It also reads the last block under the schedule rule, where B.75's increment was not established, with the beam inputs.

**Status.** Exploratory on used data: rows, targets, designs, learners, rules and fast inputs are B.75's. The beam inputs are new; no estimate with them exists. What was looked at before this registration: the file facts of `docs/paper/nbi_files.md` (alignment, sources, notches, presence of CES values by source state) and nothing else.

## 2. Data, target, designs, rules

- Rows, targets (F, L5), designs (the units of B.73: V1, V2, V3, V4 with fits to shot 31991 and to shot 32310) and row sets as B.75: the common set for the fixed-delay rules, the S set for the schedule rule (`data/.b75/sets.npz`).
- Rules: **D10** (the previous frame visible), **D650** (history 0.65 s old), **S** (the beam-modulation schedule). History inputs from `data/.b75/inputs/`.
- Beam files: `data/nbi/s{shot}_NBI.csv`, joined to the grid rows by shot and time stamp (identical stamps; checked). An empty power is zero. Beam values of rows up to and including t are visible under every rule (actuator signals).

## 3. Beam scalars (19)

With P1 = power of NB1B + NB1C (the notched beam line), P2 = power of NB2A + NB2B + NB2C, Pt = P1 + P2, and the anchor row = the row of the newest visible CES value:

- for each of P1, P2, Pt: the value at row t, the value at the anchor row, the difference (9);
- the mean of Pt over the up to ten and over the up to fifty grid rows of the block ending at row t, and the same ending at the anchor row (at least five rows; 4);
- whether each of NB1B, NB1C, NB2A, NB2B, NB2C is above 0.1 MW at row t (5);
- the voltage of NB1B at row t, zero while it is off (1).

## 4. Estimators

Fitted per rule and target, `HistGradientBoostingRegressor(random_state=20261116)`, default settings, on (target − G), as B.75.

| Name | Inputs |
|---|---|
| T0 | ten history scalars (B.75) |
| TN | T0's inputs and the 19 beam scalars |
| TX | T0's inputs and the fourteen fast scalars (B.75) |
| TXN | TX's inputs and the 19 beam scalars |
| TEN, TBN | TN's inputs and the ECEI half or the BES half of the fast scalars (V1 and V4, rules D650 and S) |

Rows: complete for TXN (history, fast and beam scalars finite) within B.75's sets; all learners are fitted and scored on those rows. Counts are reported.

## 5. Readouts

Per design, target and rule on identical rows: RMSE of G, T0, TN, TX, TXN (and of the frame's own reading against L5); TN over T0; TXN over TN; TX over T0; TXN over TX; TEN and TBN over TN; the ratio (MSE reduction of TXN over TN) / (MSE reduction of TX over T0) with its interval; under V1 every increment by chronological block of the test discharge.

**By time since the last notch (a stratum on an input):** rows within 100 ms after an absent step, later rows, rows of blocks without one; TXN over TN and TX over T0 in each (V1, D650 and S).

**Controls (V1, D650, L5):** the beam scalars permuted over rows (within fit rows and within test rows) and TN refitted; the fast scalars permuted and TXN refitted.

**Intervals.** 95 %, 20,000 draws, shot clusters resampled (the discharge form beside it), as B.75. A statement is judged on the cluster interval.

## 6. Statements (target L5)

| Id | Statement | Expectation |
|---|---|---|
| N1 | V1, D650: TN over T0 lies above zero (the beam waveforms add to the history) | holds (+0.03 to +0.15) |
| N2 | V1, D650: TXN over TN lies above zero (the fast diagnostics add beyond the beam waveforms) | holds (+0.20 to +0.35) |
| N3 | V2, D650: TXN over TN lies above zero | holds |
| N4 | V3, D650: TXN over TN lies above zero | holds |
| N5 | V4 (fit to 31991), D650: TXN over TN lies above zero | uncertain, leaning to holds |
| N6 | V1, S: TXN over TN lies above zero | holds |
| N7 | V4 (fit to 31991), S: TXN over TN lies above zero | uncertain |
| N8 | V4 (fit to 31991), S: TN over T0 lies above zero | uncertain |
| N9 | V1, D650: TXN over TX lies above zero (the beam waveforms add on top of the fast diagnostics) | uncertain (0.00 to +0.05) |
| N10 | V1, D10: TXN over TN lies above zero | holds (+0.05 to +0.10) |
| N11 | V1, D650, beam scalars permuted: the interval of TN over T0 contains zero or lies below it | holds |

Each statement is judged separately; the tables are reported whole.

## 7. What goes into the paper (decided now)

- If N2 and N3 hold: the paper gives both increments of the fast diagnostics, over the history alone and over the history with the beam waveforms, and states that the fast diagnostics carry information on the level beyond what the actuators say. N4 and N5 decide whether a forward-in-time design is named.
- If N2 fails: the paper states that the increment of B.75 is what the beam waveforms already give, the fast diagnostics are not proposed for the delayed setting, and the proposal becomes an estimate from the late CES values and the beam waveforms.
- If N9 holds, the estimator of the paper includes the beam scalars; if not, they are a comparison row.
- N7 and N8 are reported as the state of the last block under the schedule rule, whatever they are.
- If N11 fails, the batch is not used until the cause is found.

## 8. Limits known in advance

- Exploratory on used data.
- The beam columns are samples on the 10 ms rows; the notch itself has no row. The power in the first row after a notch is a sample of the returning beam and carries the same event as the BES sample of that row.
- Only the powers and one voltage are used; currents and energy fractions are not. No beam-deposition or heating calculation is made.
- The beam waveforms are what was delivered, not what was programmed; in real time the programmed waveform is known earlier.
- One estimator per rule; the level target is built from reported values.

## 9. Checks and files

- The join: every grid row of the 641 shot files has its beam row with the same stamp (count of mismatches reported; must be zero).
- T0 and TX fitted on B.75's row sets reproduce B.75's stored estimates for units V1_r2_k0 and V4a under D650 (largest deviation at most 1e-6 eV); on the rows of this batch (if they differ) the difference is reported.
- No beam scalar of row t uses a row after t (the running means end at row t; tested by changing the beam values of later rows for 200 rows and recomputing).
- Code: `ces_prediction/experiments/b79_beam_inputs/` (may import `b75_availability/` and `b73_targets/`; edits nothing). Caches: `data/.b79/`. Output: `docs/paper/b79_beam_inputs.{md,json}`, checks `docs/paper/b79_checks.md`.
- Seed 20261179, one stream per quantity. Implementation choices left open go into `PREREGISTRATION_B79_ADDENDUM1.md`, written before any readout. This file is not edited after the first run.
