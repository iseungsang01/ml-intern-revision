# B.62 — what happens at the return of the beam: the ECEI level inside the interruption, a first-frame excess, and the rotation

Registered 2026-11-18, before any readout of this batch. CPU only, no model. Designed by the session lead from the discriminating observations named by a literature study of beam interruptions; run by a delegated agent with the code of B.58.

## 1. Why

After an interruption of the heating beam of about 10 ms the reported T_i is 0.814 of its level before in the first frame, the ECEI level 0.929, and both recover with one shape over about 40 ms (B.58). A study of the published physics (beam-ion slowing down, beam-blip experiments, edge ECE) concludes that loss of beam heating for 10 ms cannot give a fall of that size in the first frame: with classical slowing down it would be a few per cent, would keep deepening for tens of milliseconds after the beam returns, and would be larger for a longer interruption, none of which is seen (B.61 found no dose). Two readings remain, a prompt transient of the plasma at the return of the beam and an effect on the measurement in the first frame, and they predict different things in data that are in hand:

- the ECEI level inside the interruption, which the raw streams of twelve discharges resolve: a response to lost heating starts inside the interruption; a transient at the return of the beam leaves the level flat inside it and steps at the return;
- a first-frame excess of the T_i deficit over the shape it shares with ECEI: an effect on the charge-exchange value in the frame with the weakest active signal adds to the first frame only;
- the rotation of the same chord: a torque at the return of the beam moves it in the first frames.

## 2. Seen before

B.58: event-median profiles after one absent frame (T_i 0.814, 0.877, 0.936, 0.996 in rows 1, 4, 6, 10; ECEI 0.929, 0.944, 0.976, 1.000; BES 0.658, 0.951, 0.995, 1.029); the T_i deficit profile proportional to the ECEI profile with 0.009 unexplained; shape index (rows 4 to 6 over row 1) T_i 0.487, ECEI 0.559. B.61: no dose. Records: fresh rotation values in the first row after a blip about 0.975 of the level before, recovering over about six rows (an earlier readout, not re-measured here). Raw streams: 41 interruptions of one absent frame in twelve discharges, off for 10.5 ms at the median. Not seen: the ECEI level at sub-frame resolution through an interruption; a fit of the first-frame excess; the rotation profile with intervals under the definitions of B.58.

## 3. Design

**Part E — raw streams (twelve discharges, the interruptions of one absent frame found by B.58 in the raw BES).** For each interruption the times at which the beam goes off and returns are those of B.58. The ECEI level (mean of the four channels) and the BES level are averaged in bins of 1 ms from 30 ms before the beam goes off to 60 ms after it returns, as ratios to the mean of the 20 ms before the beam goes off. Quantities: the ratio in the last 3 ms before the return of the beam (inside the interruption), in the first 3 ms after the return, and in 10 to 13 ms and 20 to 23 ms after the return. Intervals: 2,000 resamples of discharges. Control: the same at pseudo-times with no interruption within 200 ms.

**Part X — first-frame excess (all 641 files, events of one absent frame as in B.58).** The event-median T_i deficit of rows 1 to 8 is fitted as c times the ECEI deficit profile plus d times an indicator of row 1. Interval for d and for d as a share of the first-row T_i deficit: 2,000 resamples of discharges.

**Part V — rotation (all files, the same events).** The event-median ratio of the fresh rotation values (rows whose value differs from the previous row's) to the median of the six rows before, rows 1 to 10, with intervals; the shape index of B.58 for the rotation.

Checks before any statement: part X reproduces the first-row ratios of B.58; part E reproduces the median off-time 10.5 ms and the count of interruptions; the control of part E stays within 0.02 of 1.

## 4. Statements

- **E1.** Inside the interruption (last 3 ms before the return of the beam) the ECEI level is below its level before: the interval of the ratio is below 1.
- **E2.** The ECEI level steps down at the return of the beam: the ratio in the first 3 ms after the return is below that in the last 3 ms before it; interval of the difference below zero.
- **E3.** The ECEI level is lower 10 to 13 ms after the return than in the first 3 ms after it (the fall deepens after the return): interval of the difference below zero.
- **X1.** The first-frame excess d is above zero.
- **V1.** The rotation in the first row after the interruption differs from its level before: the interval of the ratio excludes 1.

Reported: the binned profiles of ECEI and BES; d as a share of the first-row deficit; c; the rotation profile and its shape index; the same parts for events of two and three absent frames where there are any (not statements).

## 5. What goes into the paper

- E1 fails and E2 holds: the discussion says that the ECEI level does not fall while the beam is off and steps at its return: the change belongs to the return of the beam, not to the loss of heating. E1 holds and E3 holds: a response that starts inside the interruption and deepens is present, and the paper says that a response to lost heating is not excluded in shape, with the size argument of the literature beside it. Neither pattern: reported as undecided.
- X1 holds: the paper states the share of the first-row fall of the reported T_i that is in excess of the shape shared with ECEI, as an upper bound of what the first frame adds on the measurement side. X1 fails: no first-frame excess is shown.
- V1: stated beside them, with the literature on prompt changes of rotation at beam blips on KSTAR.
- In every case the paper does not use the fall as evidence on ion heat transport and does not name a mechanism.

## 6. Expectations

E1: ratio 0.97 to 1.01, fails (flat inside the interruption). E2: step -0.03 to -0.08, holds. E3: -0.02 to +0.02, fails. X1: d 0.01 to 0.04 of the level (a tenth to a fifth of the first-row deficit), holds or fails by a small margin. V1: ratio 0.96 to 0.99, holds. All of part E rests on about forty events of twelve discharges; wide intervals are expected.

## 7. Limits known in advance

Twelve discharges chosen for other purposes. The ECEI level is not a temperature: at the edge the emission depends on density as well. A fit of d on eight points is sensitive to the template. No beam power: which sources are switched off is not known.
