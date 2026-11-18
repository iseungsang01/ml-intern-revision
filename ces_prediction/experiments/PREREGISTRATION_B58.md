# B.58 — beam blips as a calibration of how the reported value responds to the beam, and the lag of the association away from blips

Registered 2026-11-18. CPU only, no model. Proposed by a delegated agent asked for methodology (physics and use standpoint), design fixed and registered by the session lead, run by that agent. **The definitions and thresholds below were fixed after an exploratory look at the same data (section 2); the paper marks this readout as defined after an exploratory look.** What this registration adds to the look: fixed definitions, a control that was not yet run, and the requirement that every statement also holds in each of two halves of the discharges.

## 1. Why

The increment of the estimator is carried by the beam-emission value of the present frame (B.55), and the charge-exchange value needs the same beam. The manuscript says that a change of the beam alone is neither needed nor excluded. The files hold 1,394 interruptions of the heating beam, each a known perturbation of the beam: the response of the reported T_i, of the beam emission and of the electron-cyclotron emission level, which depends on neither the beam nor the spectrometer, can be compared in shape. If the reported value followed the beam signal (a fit that is biased when the active light is weak), its deficit would recover with the beam emission, within one or two frames; if it follows the plasma, it recovers with the electron-cyclotron level. Away from blips, a route through heating would put the response of T_i after that of the beam emission by the time measured at the blips; a co-variation of the edge state would not.

## 2. Seen before (exploratory, on all 641 files)

After 1,394 events in 508 discharges (ratios to the median of the six rows before): shape index (mean deficit of rows 4 to 6 over that of row 1) T_i 0.477 [0.425, 0.540], ECEI 0.484 [0.389, 0.596], BES 0.067 [0.056, 0.078]; share of the T_i deficit profile (rows 1 to 8) left unexplained when it is proportional to the ECEI profile 0.011 [0.005, 0.032], to the BES profile 0.344 [0.302, 0.391]; in T_i deficit = a ECEI deficit + b BES deficit, b = 0.042 [-0.049, +0.133], and the part of the first-row deficit that is proportional to the BES deficit 0.014 [-0.017, +0.046] of 0.184; pseudo-events within 0.011 of 1. Event by event in rows 4 to 6 the T_i deficit is also correlated with the remaining BES deficit (rank +0.449; partial on ECEI +0.376). On quiet stretches of 464 discharges (30 ms block means, first differences, median rank correlation over discharges): T_i with the BES common mode +0.314 [+0.295, +0.338] at lag 0 and +0.047 [+0.031, +0.064] with BES leading by 30 ms; with the radial difference +0.239 and +0.017; with ECEI +0.257 and +0.019; the BES series passed through the blip response gives +0.553 and +0.500. Not seen: the positive control on windows that contain a blip; the two halves; the raw-stream add-on.

## 3. Design

All 641 files, cut population. Definitions as in the exploratory scripts (`uc_blip_shape.py`, `uc_lag_quiet.py`), copied unchanged into the batch folder with their SHA-256 in the output.

Part S, shape. Events: a step of 20 to 40 ms with six contiguous 10 ms rows before; events of one absent frame are the registered set, those of two or three are reported apart. Rows 1 to 10 after the absent frame. Quantities as ratios to the median of the six rows before: T_i, the ECEI level (mean of the four channels), the BES level (mean of nine). Control: pseudo-events at rows with 10 ms steps and no absent frame within 200 ms. Statistics on the event-median profiles with 2,000 resamples of discharges: the unexplained share of the T_i deficit profile (rows 1 to 8) under proportionality to the ECEI profile and to the BES profile; the coefficients a and b of the two-regressor fit and b times the first-row BES deficit.

Part L, lag. Quiet stretches: contiguous 10 ms rows not within 30 ms before or 150 ms after a step above 15 ms, runs of at least 30 rows. Means over blocks of three rows; rank correlation of first differences of the block means at lags -4 to +4 blocks, per discharge; the median over discharges with 2,000 resamples of discharges; null by a circular shift. X: the BES common mode (mean of the log channels), the radial difference, the ECEI level; Y: T_i. Prediction of the heating route: each discharge's BES common-mode series convolved with the normalized event-average T_i deficit profile of part S, analysed identically. Positive control: the same analysis on windows that contain a blip.

Halves: the discharges are divided once into two halves by the parity of their rank in the sorted file list.

## 4. Statements (each must hold on all discharges and in each half)

- **S1.** The unexplained share of the T_i deficit profile under proportionality to ECEI has an interval below 0.10, and under proportionality to BES an interval above 0.20.
- **S2.** The part of the first-row T_i deficit that is proportional to the BES deficit (b times the first-row BES deficit) has an interval below one third of the first-row T_i deficit.
- **S3.** Control: every control row is within 0.015 of 1 for ECEI and BES and within 0.02 for T_i.
- **L1.** For the BES common mode, the ratio of the correlation with BES leading by 30 ms to that at lag 0 has an interval below 0.5, and the prediction of the heating route an interval above 0.7.
- **L2.** The same ratio for ECEI has an interval below 0.5.
- **L3.** Positive control: on windows that contain a blip the correlation with BES leading by 30 ms is above zero and the ratio is above 0.5. If L3 fails, L1 and L2 are withdrawn as uninformative.

Reported: the profiles; the shape indices; a with its interval; the event-by-event association of the T_i deficit of rows 4 to 6 with the remaining BES deficit, plain and partial on ECEI; the bound of S2 translated into eV for quiet rows (the upper end of b times the standard deviation of the 30 ms changes of the BES common mode times the level of T_i); the partial association at lag 0 of T_i with BES given ECEI and with ECEI given BES; on the twelve discharges with raw streams, the first-row deficits against the duration of the interruption (not a statement).

## 5. What goes into the paper

- S1, S2, S3 hold: the discussion says that at the largest perturbation of the beam in the data the reported value recovers with a beam-independent emission and not with the beam signal, and gives the bound on the part that follows the signal level; the limit on the target says what remains (the event-by-event association with the remaining BES deficit, which density or a slow recovery of the beam would also give). S1 or S2 fails: the reading of the dip as a change of the plasma is withdrawn from the main text.
- L1 to L3 hold: "no mechanism is shown" becomes: away from blips the fast channels carry a co-variation of the edge state within the same 30 ms; a response through heating with the dynamics measured at blips is excluded as its main route; a bias with the signal level inside the frame is bounded by S2, with the bound in eV beside the quiet-row increment. L1 fails: a heating route is not excluded and the text says so.
- The burst panel of the events figure may give way to a panel of the three deficit profiles; that is a change of a figure of the main text and is decided with the page budget.
- "ECEI level" is used, not electron temperature: at the edge the emission may be optically grey.

## 6. Expectations

S1: 0.01 and 0.34, holds, also in each half. S2: upper end 0.046 against 0.061, holds on all discharges; in a half the interval is wider and may reach one third. S3 holds. L1: 0.15 against 0.90, holds. L2: about 0.07, holds. L3: holds, ratio 0.6 to 1.0.

## 7. Limits known in advance

Defined after a look at the same data. The ECEI level is not a temperature. With eight points the coefficient b is an upper bound, not an estimate. Slow changes of the beam over more than 100 ms are removed by the differencing and are not tested. No labels of confinement mode: "quiet stretches", not "between ELMs". Block averaging of a scattered T_i lowers every correlation; the ratio is the statistic.

## Addendum 1 (2026-11-18, written after the run; no statement is changed)

Three points that the wording above left open were settled by the agent in the code before it read any verdict, and are recorded here as run. (i) The response used for the prediction of the heating route is the event-median T_i deficit profile of part S for events of one absent frame, as section 3 says; the exploratory script had the profile of all events, and its result is reported beside the registered one. (ii) The positive control of part L uses events of one absent frame with 15 rows before and 29 after, all 10 ms apart, the absent frame entered as a missing row that opens a 30 ms block; the two other alignments of the blocks are reported and are not the verdict. (iii) "Above" in L3 means that the lower end of the interval is above the threshold, as in the other statements.

