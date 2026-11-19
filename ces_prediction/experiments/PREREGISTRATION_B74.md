# B.74 — what is in the exposure of a frame whose reported width departs: the raw streams of twelve discharges

Registered 2026-11-20, before any run of this batch. CPU only, read-only on the raw 500 kHz BES streams (with the raw ECEI and Mirnov streams beside them) of the twelve discharges that have them, and on the 10 ms files. Nothing is trained; no estimator is involved. The session lead designed it. An AI agent implements and runs it under the author's supervision. No existing file is edited.

## 1. Why

Four critiques (`docs/paper/review_loop/critique_20261120_*.md`) end in one question: the part of the reported T_i that belongs to one frame and follows the BES sample of that frame, what is it? The reviewers' unregistered checks say:

- it is gone in the following frames;
- it goes with a rise of the BES sample (29 % of frames hold 45 % of the fast-channel increment);
- the rotation from the same fit does not share it;
- on the raw streams, the reported value follows the beam-emission light averaged over its own exposure, not the single sample.

None of them looked at what the light does inside the exposure. A burst of edge light inside the exposure (an ELM or a filament) is one candidate; the KSTAR CES group names such events as distorting edge spectra (Lee et al. 2026, section 3.4), and pedestal analyses with this diagnostic drop frames that overlap an ELM (Ko et al. 2016). B.31 defined bursts on the raw BES and found the T_i value missing far more often after a burst, but it labelled a row by the bursts before its stamp, not by those inside its exposure.

This batch asks whether the one-frame departures of the reported width are frames with a burst of light inside their exposure.

**Status.** Twelve discharges, chosen earlier for other reasons; discharge 31921 was B.31's discovery discharge and enters no statement here either. The reviewers have looked at these streams for other quantities. Exploratory.

## 2. Data and definitions

- Raw BES: `data/raw_mus/.cache/s{shot}_BES.npz` (500 kHz, nine channels), with the ECEI and Mirnov caches of the same discharges. 10 ms files: `data/s{shot}.csv`, first and last row dropped, T_i above 3 keV and held repeats treated as missing, as in D1.
- **Exposure of a row.** `docs/paper/file_rules.md` places the stamp within about 2 ms of the start of a 10 ms exposure; the CES group states a 5 ms exposure (Lee et al. 2020). Two windows are carried through every readout: W10 = [stamp − 2 ms, stamp + 8 ms] and W5 = [stamp, stamp + 5 ms]. Part D measures where the association sits.
- **Burst.** The detector of B.31, unchanged (`b31_bursts/run_b31.py`: array mean in 0.2 ms bins, residual above a 50 ms running median, 6 robust standard deviations, onsets merged within 2 ms). Rows within 30 ms after a beam interruption (B.31's blip rule) are left out of every readout.
- **Excess light of a row.** Over the window: the sum of the positive residual of the binned array mean above its running median, divided by the sum of the running median.
- **Classes of a row** (from the raw stream only):
  - B: at least one burst onset inside the window;
  - P: none inside, at least one in the 10 ms before the window;
  - Q: none inside and none in the 30 ms before the window.
- **Level before** of a quantity: its median over the ten preceding rows, which must be regular 10 ms rows. **Ratio** of a row: its value over the level before. For V_rot only fresh values (different from the previous row's) are used.
- **Following level** of T_i: the median of the measured values among the next five rows, at least three measured (L5 of B.73).

## 3. Readouts

**A. Does a burst in the exposure take the value away?** By class (B, P, Q) and window: the share of rows without a usable T_i.

**B. What does the reported value do in a burst frame?** Among rows with a value, by class and window:
- the median ratio of T_i, of fresh V_rot and of the ECEI array mean (sample at the stamp);
- the median of (following level of T_i) / (level before);
- the share of rows with a ratio of T_i above 1.35, and below 0.74 (the excursion thresholds of the CXRS critique).

**C. Dose.** The median T_i ratio and the median following-level ratio by bins of the excess light (zero; then quartiles of the positive values).

**D. Where the association sits.** For windows of 1, 2 and 5 ms starting from −10 ms to +14 ms after the stamp in steps of 1 ms: the rank correlation between the T_i ratio and the ratio of the window-mean BES to the mean of the same window in the ten preceding rows, on class-Q and class-B rows together.

**E. How much of the one-frame departures are burst frames.** Among rows whose T_i ratio is above 1.35 and whose next measured value is back within 15 % of the level before: the share in class B, beside the share of class B among all rows with a value. The same for departures below 0.74. This conditions on the reported value on purpose: it describes the target; no estimator is judged.

**F. Reported beside.** Per discharge: rows, bursts, class counts. For class-B rows: the Mirnov root-mean-square over the window relative to the ten windows before, and the change of the raw ECEI array mean across the burst onset (mean of the 1 ms after minus the 1 ms before, relative).

Intervals: 95 %, the eleven discharges resampled (10,000 draws), and every statistic also per discharge. With eleven units the intervals are coarse; this is stated in the output.

## 4. Statements

| Id | Statement | Expectation |
|---|---|---|
| J1 | Window W10: the share of rows without a usable value is higher in class B than in class Q (interval of the difference above zero) | holds |
| J2 | W10, rows with a value: the median T_i ratio of class B lies above that of class Q (interval of the difference above zero) | uncertain; holds if bursts are what raises the width for one frame |
| J3 | W10: in class B the median following-level ratio does not lie above the class-Q value (the interval of the difference contains zero or lies below) | holds if J2 holds |
| J4 | W10: the median ratio of fresh V_rot in class B is within 0.03 of that of class Q (interval inside ±0.03) | holds |
| J5 | Part D, 5 ms windows: the largest rank correlation is at a start between −2 ms and +5 ms, and it exceeds the values at starts −10 ms and +10 ms (intervals of both differences above zero) | holds |
| J6 | Part E: class B is over-represented among the upward one-frame departures (interval of the share ratio above 1) | uncertain |

Each is judged separately. J2 with J3 and J4 is the reading "a burst of light in the exposure raises the reported width for that frame only, and not the shift". If J2 does not hold, the one-frame rises are not explained by bursts as B.31 defines them, and the text says so.

## 5. What goes into the paper

- If J2, J3 and J4 hold: the frame-specific rise of the reported width is described as going with bursts of edge light inside the exposure, with the share of the departures it covers (part E), and with the sentence that the files cannot say whether the ions are hotter in that frame or the fit is perturbed; the rotation table stands beside it.
- If J2 fails: the text states that the one-frame departures are not accounted for by bursts of the beam-emission light, and names what was looked at.
- J1 and part A go to the usability flag: a burst in the exposure as a cause of unusable frames.
- J5 fixes the wording on where the fast sample sits relative to the exposure.

## 6. Limits

- Eleven discharges in the statements, one campaign, selected earlier for their published context; not a sample of the 639.
- A burst is defined on the beam-emission light. It agrees with what an ELM does to that light and is not confirmed as an ELM by D-alpha, which is not in the files.
- The exposure window is not known to better than a few ms; two windows are carried and part D reports where the association peaks.
- No spectrum, fit amplitude or fit error is available. An association with bursts says in which frames the width departs, not why the fit or the plasma gives that width.

## 7. Checks and files

- The burst counts per discharge equal those of `docs/paper/b31_bursts.md`.
- The single-sample BES of each row equals the raw sample at the stamp (as `file_rules.md` section 3).
- Code: `ces_prediction/experiments/b74_exposure/`. Caches: `data/.b74/`. Output: `docs/paper/b74_exposure.{md,json}`.
- Seed 20261174.
