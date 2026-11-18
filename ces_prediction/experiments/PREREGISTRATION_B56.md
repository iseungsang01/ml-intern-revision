# B.56 — a robust state-space filter with the beam-emission array as input

Registered 2026-11-18, before any run on the confirmation set. CPU only, no network is trained or run. Proposed by a delegated agent asked for methodology (estimation standpoint), design fixed and registered by the session lead, run by that agent.

## 1. Why

The fast-channel increment of the manuscript is measured with one model family: a recurrent network against its twin. The twin is unstable between initialisations, the network has 357,570 parameters, and the reference that reads the CES history alone, the causal GP, is not of the same family as either. A filter of five parameters whose form is fixed in advance would (i) give a reference of one family with and without the fast channels, (ii) carry no initialisation, (iii) state what the fast channels add as a gain that can be quoted, and (iv) show how much of the network's margin over the causal GP is a better use of the CES history and how much is the fast channels.

## 2. Seen before

An unregistered look by the proposing agent on re-partition r2 of the pooled cross-validation only (five folds, 625 test discharges, 191,934 rows), with the specification below: KF0 over the causal GP +0.002 [-0.011, +0.016]; KFX over KF0 +0.049 [+0.041, +0.058]; the network over KFX +0.040 [+0.015, +0.061]; the twin over KF0 +0.044 [+0.032, +0.055]; Kg over KF0 +0.032 [+0.025, +0.038]; Kc over KF0 +0.042 [+0.033, +0.051]; median absolute error KFX 41.8 eV, KF0 43.3, causal GP 44.1, hold 45.3, the network 47.1; in rows not after an absent frame KFX over KF0 +0.045 [+0.037, +0.054]. Forking path, declared: the agent saw the test error of a filter with an unclipped drive once on r2 (it was worse than KF0, -0.07 [-0.29, +0.03]) and then added the clip. r2 is therefore the development set and is not part of the confirmation. Not seen: any result of these filters on r3 to r9, on the four splits or on the split by time.

## 3. Design

T_i, cut population. The specification is frozen as the development script of the look (`check_kalman_x_clip.py`), which is copied unchanged into the batch folder as `kfx_model.py`; its SHA-256 is written into the output. In words:

- State x_t = log T_i, a random walk with process variance q times the time step over 10 ms. Observation log y_t = x_t + e_t with variance r. The update is weighted by min(1, k sqrt(S)/|v|), with v the innovation and S its variance (a Huber-type robust filter).
- Drive: in the prediction step the state moves by b_c dc_t + b_g dg_t, where c is the mean of the logarithms of the nine BES channels, g the radial difference of the logarithms (column 07 minus column 05), and d the change from the previous grid row. Each change is clipped at 0.2 in absolute value. Rows with a non-positive BES channel have no drive.
- Variants: KF0 (b_c = b_g = 0; parameters q, r, k), KFX (both gains), Kc (b_c only), Kg (b_g only).
- Fit: per fold, the parameters minimise the mean squared error in eV of the one-step output on the training discharges of the fold. The output for row t uses CES values before t and BES up to t.
- Scoring: on the identical test rows as the pooled runs of B.39, with their stored errors of the network, the twin, hold, the causal GP and the offline GP; interval: a re-partition drawn, then its discharges resampled (20,000 draws for skills, 2,000 for ratios of medians).

Confirmation set: re-partitions r3 to r9 (35 folds). Secondary: the four file-level splits of the manuscript (the same filters fitted on each split's training discharges; statements read as "at least three of four"), and the split by time of B.29. Reported: the fitted parameters of every fold; the clip at 0.1 and at 0.4; r2 beside the confirmation set.

Checks before any statement is read: (a) the batch code reproduces the r2 numbers of section 2 to three decimals; (b) the rows scored are identical to those of the pooled runs (same count, same reported values); (c) no CES value of row t or later enters the output for row t (the output for row t is unchanged when the values from t on are altered).

## 4. Statements (re-partitions r3 to r9)

- **KX1.** KFX over KF0: interval above zero.
- **KX2.** Kg (radial difference only) over KF0: interval above zero.
- **KX3.** The network over KFX on the same rows: interval above zero.
- **KX4.** KF0 and the causal GP are equivalent: the 90 % interval of the skill of KF0 over the causal GP lies inside [-0.02, +0.02].
- **KX5.** The median absolute error of KFX over that of hold: interval below 1.
- **KX6.** KFX over KF0 in the rows that are not within ten rows after an absent frame: interval above zero.

Reported: the twin over KF0, KFX over the twin, Kc over KF0, KFX over Kc and over Kg; KFX over the causal GP and against the offline GP; the medians of every estimator; the same in the rows that the value rules do not flag and in the three thirds by temperature (definitions of B.54).

## 5. What goes into the paper

- KX1 holds: the filter and its increment enter the main result beside the network's increment over its twin, with the fitted gains; the methods describe the filter in a paragraph. KX1 fails: the filter is reported as a failed attempt in the supplement and the paper stays as it is.
- KX3 holds: the network stays the estimator of the paper and the filter is the transparent second estimator; the margin of the network over the filter is stated as a better use of the CES history if the twin over KF0 also has an interval above zero. KX3 fails: the filter becomes a candidate for the main estimator; this is a change of the paper's design and is put to the author before it is made.
- KX2 holds: the sentence on the beam says that the filter's increment is available from the radial difference alone, in which a beam interruption is attenuated to 0.121 of its size. KX2 fails: the filter's increment needs the common mode, and the beam caveat stays as it is.
- KX4 holds: the paper says that the causal GP behaves as a robust filter on the logarithm of the value, and KF0 is the reference of the filter family.
- KX5 holds: the limit "the median is not lowered" is rewritten: with the same inputs a five-parameter filter lowers the median below that of hold, and the network does not.
- KX6: stated beside the pooled quiet-row readout of B.54.
- The 20-parameter reading of the network in the supplement, which was read off test rows, is marked as superseded by this filter if KX1 holds.

## 6. Expectations

KX1 +0.03 to +0.06, holds. KX2 +0.02 to +0.04, holds. KX3 +0.02 to +0.06, holds. KX4 skill -0.01 to +0.01, holds. KX5 ratio 0.90 to 0.95, holds. KX6 +0.03 to +0.055, holds. Four splits: KX1 four of four; KX3 two to four of four. Gains: b_c 0.4 to 0.6, b_g 0.4 to 0.5 when both are fitted.

## 7. Limits known in advance

The specification was chosen by an agent that had seen r2; r3 to r9 re-partition the same 639 discharges, so the confirmation is of the numbers and not on untouched discharges. The clip is an ad hoc robustification and its value comes from the development set. The filter reads two summaries of the BES array and no ECEI. The pooled folds are those on which the recurrent design was adopted.

## Addendum 1 (2026-11-18, before any run on the confirmation set, the four splits or the split by time)

The agent that wrote the development script compared it with this registration before running anything and found that section 3 misdescribes it in five places. No statement had been read. The specification is corrected here to what was actually run on r2.

1. The development script `check_kalman_x_clip.py` does all its work at top level and cannot be imported without repeating the r2 run. It is kept in the batch folder byte for byte as `kfx_model.py` with its SHA-256 recorded; the batch executes only its definition part (the source before the line `rng = np.random.default_rng(3)`: data loading, the filter, the loss, the mapping of rows) and reimplements the loop over folds. The fit of the gains with a fixed clip is that of the second development script `check_kfx_components.py`, kept likewise as `kfx_components.py`.
2. Clip. The development script did not fix the clip: per fold it fitted the gains at each clip in {0.02, 0.05, 0.1, 0.2, none} and kept the one with the lowest mean squared error on the training discharges; on r2 every fold chose 0.2. This registration fixes the clip at 0.2 for every fold, a simpler rule that gives the same filter on r2. The values 0.1 and 0.4 are refits and are reported only.
3. Fit in two stages, not jointly. First q, r and k are fitted with no drive (KF0; Nelder-Mead from log q = log 0.002, log r = log 0.01, log k = log 2, at most 250 iterations). Then the gains of Kc, Kg and KFX are fitted with q, r and k held at the values of KF0 (Nelder-Mead from zero, at most 300 iterations). A joint fit was tried on r2 and abandoned there because it degenerated in three of five folds; this is a second forking path on the development set and is declared with the first.
4. Rows with a non-positive BES channel (117 of 245,925 grid rows; 23 with all nine non-positive, 104 with one in column 05 or 07). The code, which is the specification: c is the mean of the logarithms of the positive channels and is undefined, so that the row has no drive from c, only when all nine are non-positive; g takes the logarithm of a value clipped below at 1e-9, so a non-positive channel in column 05 or 07 gives a large change of g that the clip of 0.2 caps. The sentence of section 3 that such rows have no drive is withdrawn.
5. Check (a) compares the point estimates of section 2 to three decimals. The intervals of section 2 came from 4,000 resamples of the discharges of r2; the batch recomputes the r2 intervals with 20,000 resamples of its discharges (one re-partition has no first level to draw) and reports them.

Fixed as well: the training discharges of a fold are all discharges that are not in its test fold (those the network used for training and for early stopping), as in the development script; on the four splits likewise every discharge that is not a test discharge of the split.

## Addendum 2 (2026-11-18, written after the run; no statement is changed)

One deviation, found and corrected by the agent during the run and recorded here. The frozen script locates a scored row in its data file as the first grid value within 0.01 eV of the stored value; where two values of a discharge differ by less than that, 45 of the 1.69 million pooled rows were placed one or more grid rows early. The batch locates each row as the closest grid value within 0.0003 eV instead, which agrees with the independent location of B.54 on every pooled row. The location decides only which row's filter output is scored; the filter and its fit do not use it, and check (a) gives the same digits either way. On the four splits and the split by time no row moved. Check (b) as registered had passed before the correction.

