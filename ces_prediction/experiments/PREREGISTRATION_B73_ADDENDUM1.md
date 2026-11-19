# B.73 — addendum 1 (session lead): strata defined on the inputs

Written 2026-11-20, after the registration (commit ee30140) and before any readout of the batch: the implementing agent had not yet computed any estimate. Reason: a fourth critique (`docs/paper/review_loop/critique_20261120_cxrs.md`) arrived after the registration. Its unregistered checks split the fast-channel increment by the sign of the departure of the BES sample and by the position of a row relative to the beam-off frames, and compare the reported width with the rotation from the same fit. These strata use inputs and the time structure only, which section 6 of the registration allows. Their thresholds were set by the reviewer after looking at the data; they are taken over unchanged and this is stated.

The implementing agent's own choices go into addendum 2.

## A1. Strata by the BES sample relative to its recent level

- r_B(t) = the BES array mean of row t divided by the median of the BES array mean over the ten preceding grid rows. Defined when those ten rows are regular 10 ms rows of the same block (no absent frame in the last 100 ms).
- Classes: below 0.9; 0.9 to 0.97; 0.97 to 1.03; above 1.03. Rows without r_B form a fifth class, reported.
- Per class, under V1: the share of rows; the increment against F and against L5 for TX over T0 (each fitted on all fit rows of its target, scored by class), for KX over K0, and for the stored pairs (two-stage estimator over its twin, network over twin) as built and with the gain fitted; the share of the F-reduction that the class holds.
- The same for TX over T0 under V4.

## A2. Strata by the position of a row relative to absent frames

- An absent frame is a step of 15 to 45 ms inside a block (the definition of B.54).
- Classes of a scored row: an absent frame both earlier and later in its discharge file; only later (before the first); only earlier (after the last); a discharge without any absent frame.
- Per class, under V1: the readouts of A1.

## A3. The rotation from the same fit (reported; no estimator involved)

- Rows with ten regular rows before them. For T_i, for a fresh V_rot (a value that differs from the previous row's, so not a held repeat) and for the ECEI array mean: the median ratio to the median of the ten preceding rows, by class of r_B (0.40–0.60, 0.60–0.80, 0.80–0.90, 0.90–0.97, 0.97–1.03, 1.03–1.10, 1.10–1.20, 1.20–1.40, above 1.40).
- The log-log slopes of the T_i ratio and of the V_rot ratio on r_B, on the rows that have both, with a 95 % interval (discharges resampled).

## A4. Rows exposed without beam emission (sensitivity)

- A row whose every BES channel is below 0.4 of that channel's median over the ten preceding rows is taken as exposed without the beam that BES views.
- Reported: the number of such rows, and how many carry a T_i value and are D1 rows.
- The increments of Q2 and of V1 with target F are given with these rows removed from the scored rows and from the fit rows of T0 and TX. The history inputs are not rebuilt.

## A5. Statements added

| Id | Statement | Expectation |
|---|---|---|
| Q11 | V1, class r_B above 1.03: the increment (TX over T0, each fitted to its target) lies above zero against F and does not lie above zero against L5 | holds |
| Q12 | V1, class r_B below 0.9: the increment against L5 lies above zero | holds (+0.05 to +0.15) |
| Q13 | The log-log slope of T_i on r_B lies above 0.5, and the interval of the slope of fresh V_rot lies within ±0.1 | holds |

All on the cluster interval, except Q13 (discharges resampled, as it involves no estimator).

## A6. What these add to the paper rules of section 8

- If Q11 holds, the text does not cite the rise of the reported value with a rise of the BES sample as information on the ion temperature. It is reported as a property of the reported width in that frame.
- If Q12 holds, the lasting part is stated with its share of frames, and with the sentence that a fall that lasts is not shown to be a fall of temperature (cooling, displacement and a low reading on a weak beam are not separated by the files).
- Q13 is printed beside them as the rotation witness, with the limit that 54 % of the rotation values repeat the previous one and only fresh values enter.
