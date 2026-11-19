# B.74 — Addendum 1: implementation choices

Written 2026-11-20 before any readout of B.74: no class has been counted, no ratio, share, median or rank correlation computed. Facts and choices only; no registered definition is changed. Code: `ces_prediction/experiments/b74_exposure/run_b74.py` (stages `check`, `prep`, `readout`); caches `data/.b74/`; output `docs/paper/b74_exposure.{md,json}`. Section 9 is filled after the checks and the per-discharge row tables, still before the readout.

## 1. Rows of the 10 ms files

- **Reading.** `data/s{shot}.csv` is read with the `csv` module (no pandas). The first row (t = 0, no CES value) and the last row (a step of at least 0.5 s, no CES value) are dropped by the rule of `analyze_file_rules.load`. A row with a missing fast channel would be dropped as in `seq_data.load_grid_files`; the count is reported.
- **T_i as in D1.** Values above 3000 eV are missing; then a held repeat is missing: a measured value equal to the previous measured value of the same block, a block being cut at a time step of 0.5 s or more. This is the comparison of `KSTAR_CES_Dataset._stuck_repeat_mask` on the same float64 values, applied after the 3 keV cut as in `load_grid_files`. It is re-implemented, not imported: `seq_data` imports `dataset`, which imports torch, and `load_grid_files` reads all 641 files with pandas, which the loaded machine cannot afford.
- **Fresh V_rot.** A measured V_rot that differs from the V_rot of the previous grid row of the file; a missing value in the previous row counts as different. No other rule is applied to V_rot.
- **ECEI and BES samples of a row.** The array means of the four `ECEI_*` and of the nine `BES_*` columns of the file. Section 7 checks that the BES values are raw samples at the stamp.

## 2. Raw time base and windows

- Sample i of a raw stream lies at t0 + i / fs (`meta` of the cache, per diagnostic; BES of 31747 is at 1 MHz). The stamps lie on the 10 ms grid; the BES t0 lies 0 or 1 µs after it, so a stamp falls on a sample or between two samples. No interpolation is done.
- A window is half-open, [a, b). Its samples are those with time in [a, b): indices ceil((a − t0) fs − 10⁻⁶) to ceil((b − t0) fs − 10⁻⁶).
- **W10** = [t − 2 ms, t + 8 ms), **W5** = [t, t + 5 ms).
- **Burst onsets** are the times returned by `run_b31.bursts_and_blips` (centres of 0.2 ms bins). An onset lies in a window when a ≤ onset < b.
- **Excess light.** `bursts_and_blips` returns neither the binned array mean nor its running median. Both are recomputed with the constants imported from `run_b31` (`BIN_S`, `MED_BINS`, `K_SIGMA`, `MERGE_S`) and the same operations. The onsets derived from the recomputed residual must equal those of the imported function on every discharge; otherwise the run stops. A 0.2 ms bin belongs to a window when its centre lies in [a, b). Excess light = Σ max(residual, 0) / Σ running median over those bins.

## 3. Which rows enter

- **Raw span.** A row enters a readout only if t − 200 ms ≥ t0 and t + 50 ms ≤ t_end of the raw BES stream. The 200 ms is B.31's rule for a labelled row. The 50 ms covers the latest Part-D window end (+19 ms) plus the 25 ms half-width of the running median. One row set serves every part.
- **Beam interruptions.** B.31's rule is applied through `run_b31.labels`: a row is left out when a 1 ms bin with the BES level below half its discharge median lies in (t − 30 ms, t]. Nothing else is left out. Two counts are reported: rows whose W10 window contains such a bin, and rows whose Part-D span [t − 10 ms, t + 19 ms) contains one, among the rows kept.
- **ECEI and Mirnov.** The readouts of part F need the raw span of their own diagnostic to cover the windows used; a row outside it has no value there (counted; Mirnov of 31873 covers about 2 s).

## 4. Classes

For a window [a, b): **B** if at least one onset lies in [a, b); **P** if none does and at least one lies in [a − 10 ms, a); **Q** if none lies in [a − 30 ms, b). A row with none in [a − 10 ms, b) and at least one in [a − 30 ms, a − 10 ms) is class **O** (other). Class O rows are counted. They enter only the readouts that take all rows: the denominators of part E and the dose bins of part C.

## 5. Ratios and the following level

- **Regular ten preceding rows.** Rows k − 10 … k exist in the file, and each of the ten steps k − 10 → k − 9, …, k − 1 → k is 10 ms within 1 ms. The step into row k is included, so the ten preceding rows are the ten preceding frames. Grid steps are 10, 20, 30, 40 ms or longer, so 1 ms separates them.
- **Level before.** T_i: median of the measured (D1) values among the ten, at least 6 of them. Fresh V_rot: median of the fresh values among the ten, at least 3 of them, and the level must exceed 20 km/s. ECEI array mean: median of the ten samples, which must exceed 0.02. The 6, 20 km/s and 0.02 are the conventions of the CXRS critique's dose table (`c18_dose_table.py`), whose excursion thresholds the registration takes over. The 3 for V_rot is the minimum of B.73's history scalars; about one grid row in three carries a fresh rotation value (`PROJECT_KNOWLEDGE.md`, held values: 35.2 %).
- **Ratio** = value of the row / level before. For V_rot only a fresh value of the row is used.
- **Following level (L5).** The later rows j = 1 … 5 are the grid rows of the same file and block whose time is 10 j ms after t within 3 ms (B.73, `common.later_values`). L5 is the median of their measured D1 T_i values, with at least 3 measured. Following-level ratio = L5 / level before of T_i.
- **Next measured value (part E).** The first measured D1 T_i among those later rows j = 1 … 5. A row without one is not counted as a departure.

## 6. Populations and statistics by part

- **A.** All rows of section 3 in class B, P or Q of the window. A row has a usable T_i when its D1 value is measured. No regular-row requirement applies.
- **B.** Rows of section 3 with a T_i ratio: a measured value, regular ten preceding rows, level before defined. Every ratio of part B (T_i, fresh V_rot, ECEI, following level) is read on this population where it is defined. So J4's V_rot ratio is read in the same frames as the T_i ratio. Shares above 1.35 and below 0.74 are shares of these rows.
- **C.** The part-B population, all classes together, per window. Bins: excess light equal to 0, then the quartiles of the positive values. The quartiles are taken over the eleven discharges pooled and fixed on the full sample; they are not redrawn per resample. Note written before the readout: as defined, excess light is 0 only if no 0.2 ms bin of the window lies above the running median. In a stationary stream about half of them do, so the zero bin is expected to be nearly empty. It is reported as defined.
- **D.** Windows [t + s, t + s + L) for s = −10 … +14 ms and L = 1, 2, 5 ms, applied to the BES array mean (mean of the nine raw channels per sample). The reference is the mean of the ten window means of the same window placed at the stamps of the ten preceding rows. Population: the part-B population in class Q or B of **W10**; the same with the classes of W5 is reported beside. Statistic: the Spearman rank correlation (average ranks) between the T_i ratio and this BES ratio over the pooled rows of the eleven discharges. **J5:** s* is the start with the largest point estimate among the 25 starts of the 5 ms windows. J5 holds if −2 ≤ s* ≤ +5 ms and the intervals of ρ(s*) − ρ(−10) and ρ(s*) − ρ(+10) both lie above zero, with s* fixed and the same draws. The share of draws whose own maximum lies in [−2, +5] is reported beside.
- **E.** Upward one-frame departure: T_i ratio > 1.35 and |next measured value / level before − 1| ≤ 0.15. Downward: T_i ratio < 0.74 and the same return. The statistic is (share of class B among the departures) / (share of class B among the whole part-B population, all classes). **J6** is read on W10; W5 is beside.
- **F.** Per discharge: rows, bursts over the whole raw stream and inside the span of section 3, class counts per window (part-A rows). For class-B rows of part A, with class Q beside as the reference:
  - **Mirnov.** For each channel (`MC1T03`, `MC1T16`), the RMS over the window after subtracting the window mean, divided by the root of the mean squared RMS of the ten windows at the stamps of the ten preceding rows. The row's value is the mean of the two channel ratios. It is computed from 0.1 ms bin sums of x and x².
  - **ECEI.** The raw ECEI array mean (four channels) over [onset, onset + 1 ms) minus that over [onset − 1 ms, onset), divided by the latter (which must exceed 0.02). The onset is the first one inside the window. For class-Q rows, the same is read at the stamp.
- **Statements J1–J4** read W10: J1 on part A, J2–J4 on part B. J1 holds if the interval of share(B) − share(Q) lies above zero; J2 if that of the median-ratio difference B − Q lies above zero; J3 if the lower end of the following-level difference B − Q is at most zero; J4 if the interval of the fresh-V_rot difference B − Q lies inside [−0.03, +0.03]. J6 holds if the interval of the share ratio lies above 1.

## 7. Intervals

- Weights: `numpy.random.default_rng(20261174).multinomial(11, [1/11]*11, size=10000)` over the eleven discharges in shot order. One weight matrix serves every statistic, so differences and ratios are formed draw by draw.
- A pooled statistic weights each row by its discharge's multiplicity in the draw. Shares are ratios of weighted counts. Medians are medians of the expanded sample (the mean of the two middle values when the count is even). Rank correlations use the average ranks of the expanded sample, computed from cumulative weights. In the resamples, ties between distinct rows are broken by order; the point estimate uses exact average ranks. The number of tied values is reported.
- Interval: the 2.5 % and 97.5 % quantiles of the draws. A draw in which a statistic is undefined is dropped; the number of draws used is reported.
- Per discharge: every statistic on each discharge alone (point only). Discharge 31921 is treated the same way, reported apart, and enters no pooled number and no statement.

## 8. Checks (registration section 7), run before the row tables

1. The burst counts of `bursts_and_blips` equal `docs/paper/b31_bursts.json` on all twelve discharges, and the onsets from the recomputed residual (section 2) equal the imported ones. Otherwise stop.
2. For every grid row inside the raw BES span, the nine BES values of the file equal one raw sample at offset −1, 0 or +1 from rint((t − t0) fs), all channels within 0.6 × 10⁻⁵ (five printed decimals; `analyze_file_rules`). Stop if the share is below 0.99 on any discharge.

## 9. Facts from the checks and row tables (filled before the readout)

Stages `check` and `prep` have run; no class has been tallied against a value, and no ratio, share, median or correlation has been computed.

- **Check 1 passed.** The burst counts are 8, 16, 0, 51, 66, 52, 7, 63, 54, 13, 30, 53 (31097 … 32097 in shot order), equal to `b31_bursts.json`. The onsets from the recomputed residual equal the imported ones on all twelve discharges.
- **Check 2 passed.** The share of rows whose nine BES values equal a raw sample at the stamp is 1.000 on all twelve.
- **Rows** (eleven discharges; 31921 in brackets):
  - 6,308 grid rows (508); 5,733 inside the raw span (483).
  - 224 left out by the beam-interruption rule (9), 88 of them in 31747, whose beam emission ends at 8.0 s. That leaves a row set of 5,509 (474), of which 5,220 have regular ten preceding rows (453).
  - **Part-B population** (rows with a T_i ratio): 4,790 (378). Of these, 4,736 have the following level (367).
- **Data treatment.** First and last rows were dropped in every file. No row was dropped for a missing fast channel. 9 T_i values above 3 keV were set missing (1), and no held T_i repeat was found.
- **Fresh V_rot ratio.** Defined on 1,029 rows of the part-B population, in eight of the eleven discharges: 31114 209, 31357 18, 31359 131, 31747 97, 31902 356, 32027 3, 32092 17, 32097 198. It is absent in 31097, 31873 and 31923. Their files carry one rotation value repeated through the discharge (fresh values: 1, 0, 0). In 31921 it is defined on 189 rows. J4 therefore rests on eight discharges.
- **Beam-interruption bins among the kept rows.** 11 rows have such a bin inside W10, and 97 inside the part-D span [t − 10 ms, t + 19 ms).
- **Spans.** The raw Mirnov stream of 31873 covers 5.412 to 7.412 s: 169 of its 673 regular rows have a Mirnov value. The raw ECEI of 31873 ends at 12.900 s, the BES at 13.082 s. All other ECEI and Mirnov spans equal the BES span. The Mirnov 0.1 ms bins align with the stamps exactly.
