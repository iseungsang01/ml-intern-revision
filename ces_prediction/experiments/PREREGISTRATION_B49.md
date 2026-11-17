# B.49 — the average of three initialisations as the estimator

Registered 2026-11-18, after arm A of B.39 was read and its statements S1 and S2 failed, and before any of the readouts below was computed. CPU only; nothing is trained, frozen networks are run forward. Designed by the session lead and executed by a delegated agent from this document.

## 1. Why

In B.39 the causal variant has a lower squared error than the causal GP in all 12 trainings on the four splits (+0.058 to +0.144), but the registered interval, which draws one of the three initialisations as well as the test discharges, is above zero on two of the four splits only: the spread between initialisations (0.006 to 0.067) and a few discharges decide the squared error of single trainings. An estimator for use need not be a single training. The average of the estimates of several initialisations is one estimator, costs a multiple of 0.5 ms per step, and removes most of the spread that the interval of S1 carries. Whether that estimator is separated from the causal GP and from the same average without the fast channels is a statement about a different estimator than S1 and S2, registered here because they failed; it is reported as such.

## 2. Seen before

B.39 arm A (`docs/paper/b39_primary.md`, `docs/paper/b39_inits.md`): S1 +0.067 [−0.014, +0.120], +0.100 [+0.068, +0.129], +0.141 [+0.101, +0.182], +0.098 [−0.017, +0.162]; S2 +0.071 [−0.012, +0.146], +0.041 [+0.006, +0.082], +0.089 [+0.037, +0.145], +0.062 [−0.046, +0.147]. With the squared error (not the estimate) averaged over the three initialisations and the discharges resampled: over the causal GP +0.067 [−0.002, +0.120], +0.100 [+0.068, +0.126], +0.141 [+0.103, +0.178], +0.098 [+0.036, +0.150]; over the twin +0.071 [+0.008, +0.118], +0.041 [+0.017, +0.065], +0.089 [+0.057, +0.124], +0.062 [+0.027, +0.099]. An earlier batch with whole-discharge standardization and four initialisations found a gain of about +0.02 from averaging. Not seen: any error of an averaged estimate of the causal variant or of its twin.

## 3. Design

T_i, cut population, the four file-level splits, the test rows of `eval_seq` (the rows of `comparison_errors_test.npz`). Members: the three initialisations of the causal variant of B.39 (`data/.b39_v2_cut_s{seed}_i{init}`, `run_b39.inits(seed)`) and the three of the history-only twin (`run_b39.twin_dir(seed, init)`). Estimator E: at every row the mean of the three T_i estimates of the causal variant. Twin average E_h: the same for the twin. Hold, the causal GP, PCHIP and the offline GP as stored.

Check before the readout (the run stops if it fails): for every member the absolute error recomputed on the CPU agrees with the stored one within 0.5 eV on at least 99 % of the scored rows.

Intervals: 95 %, 2,000 resamples of the test discharges. The average is one estimator, so no initialisation is drawn; the dependence on which trainings enter it is reported through the three averages of two members.

## 4. Statements

Four splits; a statement holds if it holds on at least three.

- **E1.** The average E has skill over the causal GP with the interval above zero.
- **E2.** E has skill over the twin average E_h with the interval above zero.
- Reported: E over hold, PCHIP and the offline GP; E over each of its members and the mean of those three (the gain from averaging); E_h over the causal GP; the three averages of two members over the causal GP and over the matching twin averages (range); the median absolute error and the 90th percentile of the absolute error of E, E_h, hold and the causal GP, and the ratio of E to the causal GP at each; the root-mean-square errors.

## 5. What goes into the paper

- E1 and E2 hold: the paper states the average of three initialisations as the estimator for use, with these numbers and the step time of three networks, beside the single-training readout of B.39 and its failed statements; the sentence says that the statements on the average were registered after those on a single training had failed.
- Either fails: stated; the four-split evidence then rests on the point estimates and the pooled readout alone.

## 6. Expectations

E over the causal GP: +0.08 to +0.16, above the mean of its members by +0.005 to +0.03 on each split; E1 holds on three or four splits (the first split is the uncertain one). E over E_h: +0.04 to +0.10; E2 holds on three or four. E_h over the causal GP: +0.02 to +0.08. The ratio of the median absolute error of E to that of the causal GP: 0.97 to 1.10.

## 7. Limits known in advance

Three members; the average of another three initialisations would differ, and that variation is seen only through the averages of two. The test rows of these splits have been read by every earlier readout, and this readout was designed after S1 and S2 failed. The members were trained independently and not as an ensemble.
