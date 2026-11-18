# B.51 — the beam-emission array split into its common mode and its differences between channels

Registered 2026-11-18, before any of the readouts below was computed. CPU only; no network is trained (part B fits gradient-boosted regressions on the CPU, as B.37 V did). Designed by the session lead and executed by a delegated agent from this document.

## 1. Why

The fast-channel increment is carried by the beam-emission (BES) channels, and the first caveat of the paper is that BES is light of the heating beam: the files record beam blips but not the beam power, so a change of the BES level may be a change of the beam or of the plasma. The nine BES channels form a three-by-three array (`BES_0105` … `BES_0307`: the first two digits index the poloidal row 01 to 03, the last two the radial column 05 to 07; the level falls from column 05 to column 07). A change of the beam multiplies all nine channels by nearly the same factor. A change of the density profile changes them differently. In the logarithm the first is a shift common to the nine channels and cancels in every difference between channels. This readout splits the array into the common mode and the differences and asks which of the two carries the association with the reported T_i and the information that the CES history lacks.

## 2. Seen before

B.39: the network that reads BES and the CES history has +0.074, +0.026, +0.105, +0.063 over the twin (four of four); the ECEI network one of four. B.45: most of the reduction of the BES network is in quiet rows. File rules: in the first row after an absent frame the BES level is 0.656 of its level before. B.43: the 200 ms changes of T_i and of the ECEI level are rank-correlated beyond a shifted null (+0.328 [+0.275, +0.370] on 400 discharges with at least ten pairs). B.37 V4: a correction layer on the causal variant fitted on validation discharges. Seen while this document was written: in one discharge (31921) the 10 ms increments of the logarithm of the nine channels are correlated at 0.70 to 0.88, higher within a column than within a row, and no BES value in 40 sampled files is zero or negative. Not seen: any quantity of the common mode or of a difference between channels.

## 3. Definitions

For a grid row, l_k is the natural logarithm of BES channel k. Rows with a non-positive channel are dropped and counted (the run stops if they exceed 2 % of the grid rows).

- Common mode c: the mean of l_k over the nine channels.
- Radial differences: g1 = mean of l over column 06 minus mean over column 05; g2 = column 07 minus column 06. Their sum g = column 07 minus column 05.
- Poloidal differences: h1 = row 02 minus row 01; h2 = row 03 minus row 02.

Row classes as in B.45 (`b45_ablation_rows/run_b45.py::Classes`): "quiet rows" have no absent frame in the ten rows before, no low beam emission in the 30 ms before, and no departure of a BES channel beyond three standard deviations in the row or the three rows before.

## 4. Part A — model-free

**Control (reported, not a statement).** At isolated absent frames (the events of B.35), the change from the mean of the six rows before the frame to the first row after it, for c, g and h1 + h2: the median over events of the absolute change of g divided by that of c. If the differences reject the beam interruption this ratio is far below 1.

**Association with T_i.** The design of B.43 S1 with its machinery (`b43_target_checks/`): 200 ms block means, the change from one block to the next, the rank correlation per discharge between the change of T_i and the change of a BES component, the median over discharges, and the same after a circular shift of one series as the null; discharges with at least ten pairs of blocks; blocks that contain an absent frame or one of the ten rows after it are left out. Interval: 2,000 resamples of the discharges.

- **A1.** The median rank correlation between the change of T_i and the change of g exceeds its shifted null, with the interval of the difference above zero.
- **A2.** The same for the common mode c.
- **A3.** The same for g after the change of c has been removed from both series by a rank regression within the discharge (partial rank correlation).
- Reported: the same for g1, g2, h1 + h2, and for c after g has been removed; the number of discharges and of block pairs.

## 5. Part B — which component carries what the CES history lacks

A second stage on the residual of the history-only twin, with the machinery and the population of B.37 V (`b37_plan/run_b37.py::part_v`: fitted on the validation discharges of the split, applied once to its test rows; gradient-boosted regression with the settings of V4). The row table is that of B.48 (`data/.b48/rows_cut_s{seed}.npz`, k = 1); the target of the regression is y minus the estimate of the twin (`hist80`).

Inputs known at the row and independent of BES ("base", in every model): the age of the last visible value, the last visible value, the estimate of the twin, the estimate of the causal GP, the difference between hold and the twin, the time step.

For a signal s the three features are: s at the row minus the mean of s over the training discharges; s at the row minus its mean over the ten rows before; the same one row earlier.

- Model 0: base only.
- Model C: base and the three features of c.
- Model D: base and the three features of each of g1, g2, h1, h2.
- Model CD: base and all fifteen.

Skill is that of twin + correction; the reference is twin + correction of model 0, on the same test rows. Intervals: 2,000 resamples of the test discharges.

- **B1.** Model D has skill over model 0 with the interval above zero.
- **B2.** Model C has skill over model 0 with the interval above zero.
- Reported: model CD over model 0, over C and over D; each of the four models over the twin itself and over the causal GP; the same three skills restricted to quiet rows; the ratio of the squared-error reduction of model D (against model 0) to that of model CD; for comparison the stored skill of the BES network of B.39 over the twin on the rows both score.

Four splits; a statement holds if it holds on at least three.

## 6. What goes into the paper

- B1 holds (with or without B2): the caveat on the beam is restated: "a part of what the beam emission adds is carried by the differences between its channels, in which a change of the beam cancels", with the control ratio and the numbers. If A1 or A3 holds it is quoted with it.
- B1 does not hold and B2 holds: the paper states that the information is in the common mode, where the beam and the density are not separated, and that the caveat stands in full; the deciding measurement stays the beam power.
- Neither holds: stated; a second stage on 127 validation discharges is then not able to recover the increment of the network from these features, and nothing is concluded about the network. The measurement that would decide it is a network retrained on the differences alone (GPU).
- The control ratio is above 0.5: the differences do not reject a beam interruption on this array, and no sentence of the paper calls them beam-free.

## 7. Expectations

Control ratio 0.05 to 0.3. A2 holds (c against T_i: difference to the null +0.10 to +0.35); A1 uncertain (+0.00 to +0.15); A3 uncertain. Model C over model 0: +0.01 to +0.06, B2 holds on three or four splits. Model D over model 0: 0.00 to +0.04, B1 holds on one to three. Model CD over model 0: +0.02 to +0.08. The ratio of the reduction of D to that of CD: 0.2 to 0.7.

## 8. Limits known in advance

A common factor is an idealisation: beam attenuation depends on the density along the beam path, so a change of density also moves the common mode, and a change of the beam geometry would move the differences. The fast channels are single raw samples, and a difference of two of them is noisier than either. Part B is a second stage fitted on about 127 discharges per split; it shows what information the components carry beyond the twin, not what the network of B.39 uses. One initialisation of the twin. The test rows are those of every earlier readout.
