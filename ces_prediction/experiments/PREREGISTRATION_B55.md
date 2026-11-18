# B.55 — the instant of the fast-channel sample: at the time stamp, just before it, one frame earlier

Registered 2026-11-18, before any run of this batch. CPU only, no training: trained networks are applied to changed inputs. Designed by the session lead, run by a delegated agent.

## 1. Why

The estimator of a row reads the fast channels of that row of the 10 ms file (`seq/seq_data.py::build_blocks`: the fast channels of row i, the time step to the previous row, and CES values of rows before i). The file values are single raw samples taken at the time stamp of the row (`docs/paper/file_rules.md`, section 3), and the stamp lies about 2 ms after the start of the 10 ms CES exposure of the row (section 4). A real-time system computes from samples acquired before the instant at which it computes, and its fast-channel acquisition can be late as well. No estimate has been made with a fast-channel sample taken at any instant other than the stamp. Question of the author (2026-11-18): was the sample just before the instant used or the sample at it, and do the two differ.

## 2. Seen before

The file rules above (twelve discharges with raw microsecond streams: the file values equal a raw sample within one raw sample of the stamp on 6,502 of 6,502 rows for BES, 6,484 of 6,484 for ECEI and 5,917 of 5,940 for Mirnov). The replays with a late CES value (B.42, B.48) delayed the CES values and never the fast channels. The pooled skills of B.39 (over the causal GP +0.089, over the twin +0.048). Not seen: any estimate made with fast-channel samples of another instant.

## 3. Design

T_i, cut population. The 40 estimator networks of the pooled cross-validation of B.39 (`data/.b39p_v2_cut_r{r}_f{k}`), each applied to the discharges of its own test fold with the loading and standardization of `eval_seq`. The twin reads no fast channel, so its errors and those of the causal GP are the stored row errors of the same fold.

**Part R — raw streams, twelve discharges.** Inside the raw window of a discharge, the fast-channel values of every grid row are replaced by raw samples taken a time d before the sample that the file holds: d = one raw sample; 0.1, 0.5, 1, 2, 5 and 10 ms. BES, ECEI and Mirnov are moved together. Rows outside the raw window keep the file values. Scored rows: the rows of the discharge that the pooled run scored and whose stamp minus 10 ms lies inside the raw window. Each of the twelve discharges is a test discharge of one fold in each of the eight re-partitions, so eight networks estimate it. Reported beside these: the fast-channel values replaced by the mean of the raw samples in the 0.1 ms and in the 1 ms before the stamp.

**Part G — grid, all discharges.** The fast-channel values of every row are replaced by those of the previous row of its block (the first row of a block keeps its own): the fast channels are one row old, 10 ms where no frame is absent. All 639 discharges, the 40 networks, scored on the rows of the pooled readout.

Quantities, per variant: (i) the root-mean-square change of the estimate in eV against the estimate made with the file values, beside the root-mean-square error of the estimator on the same rows; (ii) the change skill, 1 minus the sum of squared errors with the moved samples over that with the file values, on the same rows; (iii) the skill over the causal GP and over the twin, beside the same with the file values.

Intervals: two-level (a re-partition is drawn, then its discharges are resampled); 2,000 draws in part R, 20,000 in part G.

Checks before any statement is read (the readout stops if one fails): (a) with the file values every network reproduces its stored squared errors row by row (largest difference of the estimate below 0.05 eV); (b) the raw sample nearest the stamp equals the file value at the five printed decimals in the shares recorded in the file rules; (c) at d = 10 ms, on rows whose previous row is 10 ms earlier, the raw sample equals the file value of the previous row in the same shares; (d) part G with no shift reproduces P1 and P2 of B.39 to three decimals.

## 4. Statements

Equivalence margin: 0.01 in skill, a fifth of the pooled fast-channel increment. "Equivalent" means that the 95 % interval of the change skill lies inside [-0.01, +0.01].

- **R1.** With the raw sample one step before the one the file holds, the estimate is equivalent to that with the file values.
- **R2.** With the sample 0.5 ms earlier (the time of one step of the estimator on a CPU), equivalent.
- **R3.** With the sample 2 ms earlier (the start of the exposure), equivalent.
- **G1.** With the fast channels one row old, the estimator has skill over the twin with the interval above zero.
- **G2.** With the fast channels one row old, the estimate is equivalent to that with the file values.

Reported: every other d and the two means; the root-mean-square changes; the skills over the causal GP and over the twin for every variant; the values of each of the twelve discharges; in part G, the change skill and the skill over the twin in rows 1 to 3 after an absent frame.

## 5. What goes into the paper

- R1 to R3 hold: the methods say that the fast-channel input of a row is a single raw sample at its time stamp, about 2 ms after the start of the exposure, and that samples taken up to 2 ms earlier give the same error, with the numbers; a paragraph and a table in the supplement.
- One of R1 to R3 fails: stated as a limit. The estimate then depends on the instant of the sample inside the frame, and the real-time statement is conditional on sampling at the stamp. The measurement that would remove the limit is named: training with sample times jittered inside the frame, which needs raw streams for the training discharges, or training on averaged samples.
- G1 holds and G2 fails: the text says that the increment remains and is smaller when the fast channels are one frame old, with both numbers. G1 fails: stated as a limit, the increment needs the fast channels of the present frame.
- A mean over the raw samples whose change skill has an interval above zero is named as a direction for later training and is not a claim.

## 6. Expectations

Root-mean-square change of the estimate: one raw sample 2 to 8 eV; 0.5 ms 3 to 10 eV; 2 ms 4 to 12 eV; 10 ms 10 to 30 eV, against a root-mean-square error near 200 eV. Change skill: R1 -0.003 to +0.003, holds; R2 -0.004 to +0.004, holds; R3 -0.006 to +0.004, holds (twelve discharges: the interval may reach the margin). G2: -0.030 to -0.005, fails; G1: +0.020 to +0.045, holds. In rows 1 to 3 after an absent frame, with the fast channels one row old: change skill -0.15 to -0.03.

## 7. Limits known in advance

The twelve discharges with raw streams were chosen for other purposes, not at random, and the raw windows cover part of each. The networks were trained with samples at the stamp: a difference shows how these networks respond to another instant, not what a network trained with it would reach. The start of the exposure is inferred from beam blips to within about 2 ms.

## Addendum 1 (2026-11-18, before any statement was read)

The readout stopped at check (a); checks (b), (c) and (d) had passed and no statement had been read. The limit of 0.05 eV in check (a) did not allow for the arithmetic of two devices: the stored errors were computed on the GPU and this batch runs on a CPU. Over the 40 networks and 1,536,645 rows the absolute error with the file values differs from the stored one by 0.026 eV at the median, 0.151 eV at the 99th percentile and 2.566 eV at most (345 rows above 0.5 eV; share within 0.5 eV 0.99978); the located value in the file equals the stored value within 0.00024 eV on every row, and P1 and P2 recomputed from the CPU errors equal those of B.39 within 0.00002. The cause of the difference was not tested (a rerun on the GPU was not allowed while it is in use).

Check (a) is fixed as the rule that B.49 used for the same comparison: for every network, the absolute error with the file values lies within 0.5 eV of the stored one on at least 99 % of its rows. Check (d) is unchanged. Every change of an estimate and every change skill of this batch compares two estimates computed on the CPU, so the difference between the devices does not enter them.

Details that the registration left open, fixed as implemented before any statement was read: the raw window of a discharge is the intersection of the windows of its three raw streams; the same rows are replaced in every variant, those with raw samples back to 10 ms plus one raw step before the stamp; the sample that the file holds is found by matching all channels of a diagnostic at five decimals at the offsets 0, -1 and +1 raw steps from the stamp, and where none matches (Mirnov only) the nearest sample is taken; a mean over 0.1 or 1 ms is the mean of the raw samples that end at the held sample; in part G the standardized fast channels are moved by one row within a block; check (c) passes if the share is not more than 0.0005 below the recorded one; "near 200 eV" in the expectations means 150 to 250 eV.

## Addendum 2 (2026-11-18): part D, by diagnostic — registered after parts R and G were read, before any run of part D

Seen in parts R and G: no statement holds. With all three diagnostics moved, the estimate changes by 12.6 eV (root mean square) at one raw sample and by 39 to 57 eV from 0.1 to 5 ms, while the change skill up to 2 ms is -0.005 to -0.023 with intervals that include zero and are wider than the margin; with the fast channels one row old the estimator has -0.027 [-0.050, -0.008] against the twin. The inputs do not move alike: at 0.1 ms and beyond the Mirnov channels change by 2.3 to 2.5 training standard deviations (a snapshot of a fast oscillation), BES by 0.10 to 0.24 and ECEI by 0.09 to 0.16. B.39 found the estimator not separated from a network without the Mirnov channels on any of four splits. Parts R and G moved the three diagnostics together, so they do not say which one makes the estimate depend on the instant, nor whose present-frame value carries the increment.

Design: the same 40 networks, rows, quantities and intervals as parts R and G.

- On the twelve discharges, at d = one raw sample, 0.5 ms and 2 ms: (i) only the Mirnov channels moved; (ii) BES and ECEI moved, the Mirnov channels at the file values.
- On all discharges: one row old for (i) only BES, (ii) only ECEI, (iii) only Mirnov; and (iv) the Mirnov channels set to their training mean at every row (zero after standardization), BES and ECEI at the file values.

Statements:

- **D1.** At d = 0.5 ms the root-mean-square change of the estimate with only Mirnov moved is larger than with BES and ECEI moved: the interval of the ratio is above 1.
- **D2.** With BES and ECEI moved 2 ms earlier and Mirnov at the file values, the estimate is equivalent to that with the file values (margin 0.01).
- **D3.** With only BES one row old, the change skill has an interval below zero.
- **D4.** With only ECEI one row old, equivalent. **D5.** With only Mirnov one row old, equivalent.
- **D6.** With the Mirnov channels at their training mean, the estimator has skill over the twin with the interval above zero. **D7.** That estimate is equivalent to the one with the file values.

Reported: every quantity of parts R and G for every variant; for the variants on all discharges also the rows 1 to 3 after an absent frame.

What follows:

- D1 and D6 hold: the limit on the instant of the sample names the Mirnov channels as its source and says that the increment does not need them; the next training batch trains the estimator without the Mirnov channels in the pooled cross-validation and repeats R1 to R3 on it. D7 holding as well: the paper says that the present estimator can be run with the Mirnov inputs held at their mean, with the numbers.
- D1 fails: the dependence on the instant is in BES or ECEI, and the remedy named is averaging the raw samples, which needs raw streams for the training discharges.
- D3 holds and D4, D5 hold: the paper says that the increment needs the BES value of the present frame. D3 fails: the loss in part G is not attributed to one diagnostic.
- Failed statements are reported as failed.

Expectations: D1 ratio 2 to 6, holds. D2 change skill -0.010 to +0.005, interval about 0.02 wide on each side, does not hold. D3 change skill -0.09 to -0.04, holds. D4 -0.010 to +0.002, holds. D5 -0.015 to +0.005, does not hold. D6 skill over the twin +0.030 to +0.055, holds. D7 change skill -0.020 to +0.010, does not hold.

Limit added: a network that was trained with the Mirnov channels and is given their mean is not a network trained without them.
