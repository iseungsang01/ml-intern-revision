# B.46 — the reported T_i and the last beam-off frame: a step at the frame, and the frame-to-frame component by the age of the frame

Registered 2026-11-18, before any of the readouts below was computed. CPU only; nothing is trained. Designed by the session lead and executed by a delegated agent from this document.

## 1. Why

The paper estimates the T_i value that the KSTAR CES analysis reports. Its data section says that an offset constant in time changes no skill and that "a bias that varies within a discharge would". No measurement of such a bias is in hand. The CES analysis takes its background from frames in which the beam is off; in the 10 ms files a beam-off frame is an absent frame (a time step of 20 to 40 ms). If a quantity that the analysis refreshes at a beam-off frame moved the reported value, two things would be seen in the files: (i) the reported value would change across an absent frame by more than it changes over the same span elsewhere, and (ii) its frame-to-frame independent component would depend on the time since the last absent frame. The ECEI level, which uses neither the beam nor the spectrometer, is the control for changes of the plasma. Either outcome gives the paper a number: the size of such a component, or a bound on it.

## 2. Seen before

The response to a beam blip: in the first row after an absent frame T_i is 0.819 and the ECEI level 0.926 of the level before, both return over about 90 ms (integral recovery times 41.0 and 40.2 ms). The frame-to-frame independent component (B.40): 39.1 to 42.8 eV on a typical row, growing with the level. B.43: the 200 ms changes of T_i and of the ECEI level have a rank correlation of +0.268. B.44: the two margins of the paper do not depend on the number of absent frames per second of a discharge. Not seen: the level of T_i before against after an absent frame once the recovery is over; the independent component by the time since the last absent frame; the number of absent frames in the data set by discharge.

## 3. Data

All 641 discharges, 10 ms files; the first and the last row of a file are not grid rows. T_i with held values and values above 3 keV removed (`load_grid_files(DATA, True, 3000.0)`, as in B.40 and B.44). ECEI level: mean of the four channels. A segment ends at a time step of 0.5 s or more. An absent frame: a time step between 15 and 45 ms inside a segment; `tb` is the stamp of the last row before it and `ta` that of the first row after it.

## 4. Part J — a step across an absent frame

Events: absent frames with no other absent frame in the 400 ms before `tb` or the 400 ms after `ta`, in one segment. Window before: stamps in (tb − 200 ms, tb]. Window after: stamps in [ta + 150 ms, ta + 350 ms), which begins after the recovery. Each window needs at least 10 measured T_i rows. Step of an event: the natural logarithm of the mean T_i of the window after over that of the window before; for the ECEI level the same with the rows of the two windows (events whose ECEI window mean is below 5 % of the discharge median of the absolute ECEI level are dropped for ECEI and counted).

Controls: in the same discharges, grid stamps `tc` taken every 500 ms along a segment that have no absent frame within 600 ms on either side; window before (tc − 200 ms, tc], window after [tc + 180 ms, tc + 380 ms) (the geometry of an event with a gap of 30 ms), the same row requirement.

Spread: 1.4826 times the median absolute deviation of the steps from their median. Intervals: 95 %, 2,000 resamples of discharges (events and controls of a discharge drawn together).

- **J1.** The spread of the T_i step at absent frames over that at control stamps has an interval above 1.
- **J2.** The ratio of J1 divided by the same ratio for the ECEI level has an interval above 1.
- Reported: the numbers of events, controls and discharges; the median step at events and at controls for T_i and for the ECEI level, and their difference; the excess spread √(spread at events² − spread at controls²) in per cent of the level with its interval (zero where negative), and the upper end of that interval as the bound; the same with the window after moved to [ta + 250 ms, ta + 450 ms); the same by third of the level of T_i before the event. Not testable with fewer than 100 events or 20 discharges.

## 5. Part A — the independent component by the age of the last absent frame

Triples as in B.40: three consecutive measured rows with both steps 10 ms; the second difference y(t+1) − 2 y(t) + y(t−1). Age of a triple: the time from the last absent frame of the same segment to its middle row. Relative component of a set of triples: 1.4826 times the median absolute deviation of the second difference over √6, divided by the median T_i of the middle rows.

Per discharge with at least 30 triples at an age of 0.1 to 0.5 s and at least 30 at an age of 1.0 to 2.0 s: the ratio of the relative component at the later age to that at the earlier.

- **A1.** The median over discharges of that ratio has an interval that excludes 1 (2,000 resamples of discharges).
- Reported: the number of discharges; the pooled relative component, the share of missing T_i and the share of values above 3 keV in the age bins 0.1–0.3, 0.3–0.6, 0.6–1.0, 1.0–1.5, 1.5–3 s, above 3 s and "no absent frame before in the segment"; the distribution of the number of absent frames per discharge and of the spacing between consecutive absent frames. Not testable with fewer than 20 discharges.

## 6. What goes into the paper

- J1 fails: the data section states that the reported value shows no step across a beam-off frame beyond its change over the same span elsewhere, with the bound on the excess in per cent of the level. J1 holds and J2 fails: an excess is seen and the ECEI level shows it as well, so it is read as a change of the plasma after the interruption; stated with the size. J1 and J2 hold: the reported value has a component tied to beam-off frames that the ECEI level does not show; its size is stated and it becomes a named limit of the target.
- A1: stated with the ratio in either case; if the interval excludes 1 the sign and the size are given.

## 7. Expectations

Events: 300 to 3,000 in 100 to 400 discharges. J1: ratio 0.95 to 1.25; not established or weakly so. J2: does not hold. Excess spread below 5 % of the level. A1: ratio 0.9 to 1.1, interval including 1.

## 8. Limits known in advance

A beam interruption is a perturbation of the heating, and a change of the plasma that outlasts 150 ms would show in T_i and, in a different proportion, in the ECEI level; J2 controls for it only in part. The test sees a component that changes at a beam-off frame and none that drifts between them. How the 2022 analysis treats the background frame by frame is not documented, so a null result bounds this route of a varying bias and no other. Discharges with frequent absent frames differ from those with few; part A compares ages inside a discharge for that reason.
