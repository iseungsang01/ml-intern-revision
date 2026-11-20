# Rows in which the ECEI channels do not move

Unregistered diagnostic by the session lead, 2026-11-20 (`ces_prediction/experiments/b75_availability/lead_checks/stuck_ecei.py`). Found while checking the example discharge of the manuscript (shot 31076): its four ECEI channels are exactly constant from the start of the file to about 2 s. Input-only counts, and the stored estimates of B.75 under the random design (V1), target L5, read by a stratum on an input. No interval; nothing here is a registered statement.

A row counts as "ECEI constant" when all four ECEI channels have had the same values for at least ten rows (100 ms) within the file.

`py ces_prediction/experiments/b75_availability/lead_checks/stuck_ecei.py .`

```
(1) grid rows in which the four ECEI channels have not changed for at least ten rows (BES: the nine channels)
   block         grid rows | ECEI constant: share of rows, discharges with any, with more than half | BES constant: share of rows
   up to 31991    180560 | 0.043, 49 of 416, 4 | 0.0000
   32002-32310     42707 | 0.012, 4 of 128, 1 | 0.0000
   from 32312      22658 | 0.017, 7 of 97, 1 | 0.0000
   all            245925 | 0.035, 60 of 641, 6 | 0.0000
(2) scored rows of B.75 with a constant ECEI at row t or at the anchor row
   S_L5       rows 147837: constant at row t 0.031; at the anchor row 0.030; either 0.049 | by block (either): up to 31991 0.054, 32002-32310 0.015, from 32312 0.053
   common_L5  rows 160343: constant at row t 0.031; at the anchor row 0.034; either 0.042 | by block (either): up to 31991 0.049, 32002-32310 0.014, from 32312 0.007
(3) TX over T0 (V1, stored estimates, target L5) by whether the ECEI is constant at row t or at the anchor row
   rule S    all blocks   | ECEI moving: 140634 rows, 495 discharges, +0.334 | ECEI constant: 7203 rows, 51 discharges, +0.160
   rule S    up to 31991  | ECEI moving: 113147 rows, 347 discharges, +0.348 | ECEI constant: 6508 rows, 41 discharges, +0.161
   rule S    32002-32310  | ECEI moving: 20691 rows, 104 discharges, +0.323 | ECEI constant: 311 rows, 3 discharges, -0.121
   rule S    from 32312   | ECEI moving: 6796 rows, 44 discharges, +0.039 | ECEI constant: 384 rows, 7 discharges, +0.278
   rule D650 all blocks   | ECEI moving: 153655 rows, 559 discharges, +0.348 | ECEI constant: 6688 rows, 51 discharges, +0.059
   rule D650 up to 31991  | ECEI moving: 120981 rows, 373 discharges, +0.344 | ECEI constant: 6289 rows, 42 discharges, +0.116
   rule D650 32002-32310  | ECEI moving: 22603 rows, 112 discharges, +0.397 | ECEI constant: 324 rows, 4 discharges, -0.916
   rule D650 from 32312   | ECEI moving: 10071 rows, 74 discharges, +0.170 | ECEI constant: 75 rows
```

## Reading

1. The ECEI is constant in 0.035 of the grid rows, in 60 of the 641 discharges (in 6 of them for more than half of the rows). The BES channels are never constant in this sense. The estimators were fitted and scored with these rows in.
2. Where the ECEI is constant at the row or at the anchor row, the fast-channel increment is smaller: +0.160 against +0.334 under the beam-modulation schedule and +0.059 against +0.348 with the history 0.65 s old (all blocks). With these rows left out the overall values change little (+0.334 against +0.327; +0.348 against +0.336).
3. It does not account for the last block of shots: there the ECEI is constant in 0.017 of the grid rows, and on the rows where it moves the increment under the schedule is +0.039.

For the manuscript: the data section states that the ECEI channels are constant in part of 60 discharges and that those rows are kept; the example figure says so for its discharge.
