# B.43 — checks of the reported T_i value that the files allow

Registered 2026-11-18, before any of the readouts below was computed. CPU only; no model is trained (part D runs frozen networks forward only if the figure data do not suffice). Designed by the session lead and executed by a delegated agent from this document.

## 1. Why

The paper estimates the T_i value that the KSTAR CES analysis reports. The author asked how far that is from the true temperature and which further measurements are worth making. The random part was measured in B.40; the literature on the systematic part is in `docs/reference/ces_accuracy_literature_2026-11-18.md`. Four checks remain that need only the files and that a referee can read as evidence that the reported value follows the plasma: co-variation with a diagnostic that uses neither the beam nor the spectrometer, where the values above 3 keV sit, agreement with a published time course, and the one unexplained feature of the trace figure.

## 2. Seen before

The response to a beam blip (T_i 0.819 and ECEI 0.926 in the first row, the same integral recovery time, 41.0 and 40.2 ms). The lower bound of the correlation of 10 ms increments of the fast channels with those of T_i (+0.127). The autocorrelation times (T_i 159 ms, ECEI 147 ms). Missing T_i by fifth of the BES level (18.0 % in the lowest, 3.4 % in the highest). B.40 (the white component by BES level and by T_i level). From the literature report: the published analysis shows discharge 31923 between beam blips at 3.5 and 5.0 s with a step of the beam power from 3.5 to 4.0 MW at 4.0 s after which T_i rises, and profiles of 31921 at 5.40 and 8.05 s. Not seen: any correlation of T_i with the ECEI level on scales longer than a frame; the share of values above 3 keV by BES level; the file values of 31923 around 4.0 s; the cause of the fall of the estimate in panel (c) of the trace figure.

## 3. Parts and statements

Intervals: 95 %, 2,000 resamples of discharges unless stated. T_i with held values removed; values above 3 keV removed except in part F.

**S. Co-variation with the ECEI level on a 200 ms scale.** Inside each contiguous segment the grid is cut into blocks of 20 rows; a block is used if it has at least 10 measured T_i rows. Per discharge with at least 40 pairs of consecutive used blocks: the Spearman rank correlation between the change of the block mean of T_i and the change of the block mean of the ECEI level (mean of the four channels) from one block to the next. Null: the same with the ECEI changes shifted circularly by half the number of pairs of that discharge.

- **S1.** The median over discharges of the correlation minus the median of the null has an interval above zero.
- Reported: both medians; the same for the BES level; the same with blocks of 100 ms and of 500 ms; by third of the discharge median T_i.

**F. Where the values above 3 keV sit.** All grid rows with a T_i value, by fifth of the BES level of the row relative to the discharge median (as in B.40).

- **F1.** The share of rows above 3 keV in the lowest fifth over that in the highest fifth has an interval above 1.
- Reported: the five shares; the same for missing T_i (a check: 18.0 % and 3.4 % are to be reproduced within 0.5 percentage points); the share of the values above 3 keV within ten rows after an absent frame.

**H. A published time course (discharge 31923; descriptive, one discharge).** From the 10 ms file: the stamps of absent frames between 3.0 and 5.5 s; the mean T_i over 3.60 to 3.95 s and over 4.10 to 4.90 s.

- **H1.** An absent frame lies within 30 ms of 3.5 s and one within 30 ms of 5.0 s, and the ratio of the later mean to the earlier one has an interval above 1 (2,000 resamples of 100 ms blocks of rows).
- Reported: the 100 ms means from 3.5 to 5.0 s; for 31921 the mean and standard deviation of T_i within 50 ms of 5.40 s and of 8.05 s.

**D. The fall of the estimate in panel (c) of the trace figure (descriptive).** From `docs/paper/ppcf/fig_trace.py` and `fig_trace.json`: the discharge and the rows where the estimate of the causal variant lies below the measured range of the window; for those rows the time since the last absent frame, the mean BES departure (B.32 definition), the ECEI level relative to the 100 ms before, and the next measured T_i. No statement; the result is one sentence for the caption.

## 4. What goes into the paper

- S1 holds: the data section says that the reported value co-varies on a 200 ms scale with a diagnostic that uses neither the beam nor the spectrometer, with the number. Fails: stated.
- F1 holds: the values above 3 keV are described as sitting where the beam emission is low, as failed fits of a weak signal do. Fails: the sentence that they are failed fits rests on the literature and on their isolation only.
- H1 holds: the files reproduce the published time course of one discharge. Fails: stated.

## 5. Expectations

S1: median correlation +0.15 to +0.40 against a null within 0.03 of zero; holds. F1: ratio 2 to 8; holds. H1: both blips found; ratio 1.05 to 1.40; holds. D: the rows follow an absent frame by 10 to 30 ms.

## 6. Limits known in advance

A co-variation on 200 ms scales with ECEI shows that the reported value follows something the electrons also follow; it does not bound a bias. The ECEI level is uncalibrated and its channel means are near zero in a few discharges. One discharge for the published comparison, whose chord is not the one of the publication.
