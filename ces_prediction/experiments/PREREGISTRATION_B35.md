# B.35 — the beam blip as a repeated perturbation: does the response of T_i behave like an energy balance?

Registered 2026-11-16, before the readouts below were computed. CPU only; the 10 ms files, no model.

## 1. Why

The files hold 1,394 beam blips with six contiguous rows before them. They were used once, to check what the CES value follows (`docs/paper/file_rules.md`). They are also a perturbation repeated over 508 discharges: the heating is interrupted for one to three frames and the plasma answers. Two properties that an energy balance predicts, and that an artefact of the returning beam does not, can be read from the files alone.

- **Dose.** A longer interruption removes more heating, so the fall of T_i is deeper after two or three absent frames than after one. A fit that is disturbed only by the returning beam sees the same returning beam whatever the length.
- **Time scale.** Collisional time scales grow with temperature (as T^1.5 at fixed density), so the return of T_i is slower where T_i is higher.

## 2. Seen before

The mean response (T_i 0.819, ECEI 0.926, BES 0.656 in the first row; return over about 90 ms; deficit ratio 2.33); B.32 (the ratio by the BES departure of the first row; the counts of 20, 30 and 40 ms steps inside the three departure groups, without any T_i value by step). Not seen: the T_i ratio by the length of the interruption, any recovery time, any dependence on the level before the blip.

## 3. Events and quantities

Events: a step of 20 to 40 ms inside a segment with six contiguous 10 ms rows before it (as in `analyze_file_rules.py`), with T_i observed (at most 3 keV) in at least three of those six rows; the level before is the median of the observed values. Rows after: k = 1 to 15 (10 ms apart), used as long as they are contiguous and no further step occurs. Ratio at row k: T_i over the level before; the ECEI ratio uses the mean of the four ECEI channels relative to its median over the six rows before. Trace of a group: the median over events at each k. Deficit D(k) = 1 − ratio. Integral recovery time: 10 ms × Σ_{k=1..15} D(k) / D(1). Intervals: 95 %, 2,000 resamples of discharges.

## 4. Statements

- **N1 (dose).** The median T_i ratio at k = 1 after two or three absent frames minus that after one absent frame has an interval below zero.
- **N2 (time scale).** With the events after one absent frame divided into thirds by the T_i level before, the integral recovery time of the highest third minus that of the lowest third has an interval above zero.
- **N3 (reported).** The same two readouts for the ECEI ratio; the integral recovery times of T_i and of ECEI over all events; N1 within the highest and the lowest third.

## 5. What goes into the paper

N1 holds: the paper says that the fall of the CES value after a blip grows with the length of the interruption, which an energy balance predicts and a disturbance of the fit by the returning beam alone does not. N2 holds: the return is slower at higher temperature, as collisional time scales are. Either fails: reported as such; neither failure shows that the value is not a temperature.

## 6. Expectations

Events: 1,000 to 1,250 with one absent frame, 90 to 140 with two or three. N1: 0.82 against 0.68 to 0.78, holds. N2: recovery time 40 to 70 ms in the lowest third and 70 to 130 ms in the highest; holds. ECEI: the same sign for N1, weaker.

## 7. Limits known in advance

After two or three absent frames the first row is also later relative to the start of the interruption. Longer interruptions may be programmed in particular discharges. The level of T_i before the blip stands in for the electron temperature and says nothing of the density. A deficit that does not return within 15 rows makes the integral time a lower bound.
