# PREREGISTRATION B.31 — edge bursts in the raw beam-emission stream: where the charge-exchange gaps, the large errors and the fast-channel share sit

Registered 2026-11-16. Committed before the labelling code and before any number of section 4
exists. Stated aim (2026-11-16): keep the claim at the pedestal-top chord and carry the work on
until the rows that carry the gain can be identified. Every reviewer of the draft asked what
those rows are; the 10 ms files cannot say. The 500 kHz beam-emission (BES) stream of twelve
discharges (`data/raw_mus`, delivered 2026-10-26) can.

## 1. What has been seen (discovery, not part of the test)

Only discharge 31921 was looked at, with the detector of section 2 and nothing from any model:

- bursts per second: 13.6 (3.6-5.0 s), 0.8 (5.1-6.4 s), 0.0 (6.6-7.3 s), 30.0 (7.3-8.0 s), 34.5
  (8.05-8.6 s); the published analysis of this discharge (Heo et al. 2025) places it in FIRE mode
  at 5.40 s and in H-mode at 8.05 s;
- the measured T_i has no missing row before 7.4 s and 10 to 90 % missing rows per 100 ms after;
  in 7.6-8.4 s it moves over several rows at a time (2,085 to 103 eV in 80 ms; 1,060 to 1,182 eV
  held while the BES level rises; missing or 100 to 300 eV after a burst);
- BES level below half of its median only at 5.0, 6.501 and 8.002 s for 9 to 11 ms (beam blips);
- of discharge 31923 only the 1 ms BES level, the beam blips (5.001, 6.502 s) and the missing
  share per 100 ms were printed (no burst was counted and no model error was read).

No model error of any discharge has been broken down by burst. The B.30 row tables, which hold
the model errors, were not joined to the raw stream.

## 2. Labels (frozen; the constants are those used on 31921)

- Stream: mean of the nine BES channels of `data/raw_mus/.cache/s{shot}_BES.npz`, averaged in
  0.2 ms bins on the delivered clock.
- Burst: a run of bins above the 50 ms running median (251 bins) by more than 6 robust standard
  deviations (1.4826 x median absolute deviation of the residual over the discharge); runs whose
  starts are less than 2 ms apart are merged into the first. The burst time is the start bin.
- Beam blip: 1 ms bins with the level below half of the discharge median; rows whose preceding
  30 ms touch a blip are excluded from every statistic (reported as a count).
- Per 10 ms row at time t (the time column of the shot file): `near` = at least one burst in
  (t - 10 ms, t]; `active` = at least 3 bursts in (t - 200 ms, t]; `quiet` = no burst in
  (t - 200 ms, t]. Rows that are neither active nor quiet are `sparse`. Rows outside the raw
  stream or within 200 ms of its start are not labelled.
- The label uses the raw stream up to the row time only. It is not an input of any estimator.

## 3. Discharges and rows

- Raw set: 31097, 31114, 31357, 31359, 31747, 31873, 31902, 31921, 31923, 32027, 32092, 32097.
- Measurement statements (H31-1, H31-2) use the eleven discharges other than 31921.
- Model statements use the discharges that are TEST discharges of a file-level split, with the
  frozen B.30 row tables at k = 1 (cut population): split 42: 31114, 31873, 31902; split 1:
  31902, 31923, 32092; split 7: 31923, 32092; split 123: 32027. Discharge 31921 (TEST in splits
  42 and 1) is reported beside them and not counted in any rule. A discharge that is TEST in two
  splits enters once per split; the unit of resampling is the discharge.
- Intervals: 2,000 bootstrap resamples of discharges (seed 20261116). With five to ten
  discharges the intervals are wide; each statement is also given per discharge, and the rule of
  each statement is on the count of discharges.

## 4. Registered statements

- **H31-1 (the fit fails around a burst).** Among labelled rows, the share of rows with T_i
  missing is larger in `near` rows than in `quiet` rows in at least 80 % of the discharges
  (other than 31921) that have both kinds of row with at least 20 rows each; at least four such
  discharges are needed for the statement to be read.
- **H31-2 (the measured value moves where bursts are).** The median absolute difference of T_i
  between adjacent measured rows is larger in `active` rows than in `quiet` rows in at least
  80 % of the discharges (other than 31921) that have at least 20 such pairs of each kind; at
  least four such discharges are needed.
- **H31-3 (the hold error is there).** In the model rows, the hold RMSE of T_i is larger in
  `active` rows than in `quiet` rows in at least 80 % of the TEST discharges other than 31921
  that have both kinds (at least 30 scored rows each); at least three such discharges are needed.
- **H31-4 (the fast-channel share is there).** Pooled over those discharge-split entries, the
  part of the summed squared-error reduction of the network against its history-only twin
  (`hist80` minus `bb80`, B.30 k = 1) that falls in `active` rows exceeds the part of scored
  rows that are `active`, and the pooled skill of the network over its history-only twin in
  `active` rows has a discharge-bootstrap interval above 0.
- **H31-5 (and it is not only there).** The same skill in `quiet` rows has an interval that
  includes 0 or lies above it; if it lies above 0 the paper says that the share is present
  without bursts as well, if it includes 0 the paper says the share is established in
  burst-active phases only.

Rules for the paper. H31-1 to H31-3 hold: the paper states that the gaps, the large changes and
the large hold errors of this chord sit in phases with edge bursts, on the discharges with a raw
stream, and names them "edge bursts consistent with edge-localized modes" (no D-alpha is in the
files; the one discharge with a published mode history is 31921). H31-4 holds: the paper states
that the fast-channel share is concentrated in those phases. Any of them fails: the failed
statement is reported as failed and the mechanism stays "not identified". No statement is
extended to the 629 discharges without a raw stream.

## 5. Numerical expectations (stated so they can fail)

- Discharges with at least 20 `near` rows: 7 to 10 of 11; `active` share of labelled rows per
  discharge: 0.05 to 0.6, differing strongly between discharges.
- H31-1: missing share 0.3 to 0.7 in `near` rows against 0.0 to 0.1 in `quiet` rows; holds.
- H31-2: ratio of the median absolute differences 2 to 8; holds.
- H31-3: hold RMSE in `active` rows 2 to 5 times that in `quiet` rows; holds.
- H31-4: `active` rows are 10 to 40 % of the scored rows and carry 50 to 90 % of the reduction;
  pooled skill in `active` rows +0.05 to +0.20; holds with probability about 0.6 (few
  discharges).
- H31-5: the skill in `quiet` rows +0.00 to +0.08; interval including 0 with probability 0.5.

## 6. Disclosures

- Twelve discharges were selected for the raw delivery by an earlier rule (hires set, 93e01bb),
  not at random; five to six of them carry the model statements.
- A burst is defined on beam emission; without D-alpha it is not shown to be an edge-localized
  mode. The beam itself is on throughout except at the blips.
- Scored rows are measured rows; rows where T_i is missing cannot carry a model error, so the
  model statements describe the measured rows of burst-active phases.
- Output: `docs/paper/b31_bursts.json` / `.md`; record in `THESIS_RESULTS.md` section 8; code
  `ces_prediction/experiments/b31_bursts/run_b31.py`. CPU only.
