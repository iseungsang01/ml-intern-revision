# B.57 — yardsticks for the size of the gain: a noise floor, the gap to the offline estimate, and whether the fast-channel correction lasts into the next frame

Registered 2026-11-18, before any readout of this batch on the pooled rows other than those named in section 2. CPU only; the only networks run are the stored twins of the pooled cross-validation, forward, for their signed estimates. Proposed by two delegated agents asked for methodology (estimation standpoint; referee standpoint), design fixed and registered by the session lead, run by a delegated agent.

## 1. Why

The pooled skill over the causal GP is +0.089, and the median absolute error is not lowered. A reader has no way to judge that size: the reported value carries a scatter from frame to frame that no estimator of the present frame can remove, and the paper gives neither that floor nor the ceiling that an estimate with the next value reaches. Separately, the manuscript cannot say whether what the fast channels add describes the plasma or the fit of the frame being estimated: a beam-emission sample taken during the exposure could predict a property of that exposure's fit. The next frame is fitted independently, so a correction that describes the plasma should still be there one frame later, and one that describes the fit of frame t should not.

## 2. Seen before

Unregistered looks by the proposing agents. On the pooled rows of B.39 (eight re-partitions): a lower bound of the white variance from consecutive triples of scored rows 18,415 [15,805, 21,146] eV squared (root mean square 136 [126, 145] eV); mean squared error over that bound 1.33 [1.25, 1.44] for the network, 1.47 [1.39, 1.57] for the causal GP, 1.28 [1.22, 1.34] for the offline GP; skill relative to the removable error +0.30 [+0.19, +0.41] over the causal GP and +0.20 [+0.10, +0.31] over the twin. From the point estimates of B.39 (no interval): the network closes 0.73 of the gap between the causal and the offline GP, the twin 0.35. On the four splits (rows of B.48, one initialisation): the share of the fast-channel correction that is still there one frame later 0.90, 0.76, 0.79, 0.73; three frames later 0.85, 0.86, 0.84, 0.75; the slope of the correction against the value 0.77, 0.63, 0.96, 0.78. Not seen: the persistence on the pooled rows, for the network or for the filter of B.56; the gap closure with an interval.

## 3. Design

T_i, cut population, the rows of the pooled cross-validation of B.39. Signed estimates: the network's from the cache of B.55 (file values; the 40 networks run on a CPU), the twin's by running the 40 stored twins forward in the same way, the filters' from B.56. The interval is that of the pooled readout (a re-partition drawn, then its discharges resampled; 2,000 draws).

(a) Floor. On consecutive triples of scored rows 10 ms apart, s2 = minus the covariance of successive increments of the reported value. It is a lower bound of the mean white variance if the increments of the underlying value are not negatively correlated (the assumption registered in B.40).

(b) Removable-error skill of A over R: (MSE_R - MSE_A) / (MSE_R - s2), on the rows of the triples. Gap closure of A: (MSE_GP - MSE_A) / (MSE_GP - MSE_offline GP), on all pooled rows.

(c) Persistence. For a pair (A, B) with d_t = A_t - B_t: beta_h = cov(d_t, y_{t+h} - B_t) / var(d_t) on rows whose frames t to t+h are consecutive and scored, h = 0, 1, 3; rho_h = beta_h / beta_0. Pairs: the network and its twin (the fast-channel correction); KFX and KF0 of B.56; and, as the reference of a correction that cannot know the fit of frame t, the twin and KF0.

Checks before any statement is read: (i) the twins run on the CPU reproduce their stored squared errors by the rule of B.55 (within 0.5 eV on at least 99 % of the rows of every network); (ii) the floor and the two removable-error skills of section 2 are reproduced to three significant figures; (iii) with the reported value of frame t+1 replaced by that of frame t, rho_1 equals 1 (the arithmetic of the ratio).

## 4. Statements

- **N1.** The mean squared error of every estimator, the offline GP included, is above the floor: every interval of MSE over s2 is above 1. If N1 fails the removable-error skill is not reported.
- **N2.** The network closes more than half of the gap between the causal and the offline GP: interval above one half.
- **N3.** The fast-channel correction of the network lasts into the next frame: the interval of rho_1 for the network and its twin is above one half.
- **N4.** The same for KFX and KF0.
- **N5.** The part that does not last is specific to the frame: rho_3 minus rho_1 for the network and its twin has an interval inside [-0.15, +0.15].

Reported: the floor with its interval; MSE over the floor for every estimator; the removable-error skills over the causal GP, the twin and hold, for the network and for KFX; the gap closure of the twin and of KFX; beta_0 of every pair; rho_1 and rho_3 of the reference pair; all of (c) in the rows that are not within ten rows after an absent frame.

## 5. What goes into the paper

- N1 holds: the main result gives, beside each pooled skill, the floor and the skill relative to the removable error, and the abstract may say what share of the remaining squared error is scatter of the reported value. The names of the noise quantities are reduced to two: the typical frame-to-frame scatter of a row (39.1 to 42.8 eV, B.40) and the mean-square floor of this batch.
- N2 holds: the sentence on the offline GP reads that the estimator closes that share of the gap to an estimate that waits for the next value. N2 fails: the share is given with its interval and no statement is made.
- N3 holds: the paper says that the stated share of what the fast channels add is still there in the next, independently fitted frame, so at most the rest can be a property of the fit of the frame; and it says what this does not exclude (a bias shared by consecutive frames, through a slowly varying beam or a common background). N3 fails: the paper says that the increment may be a property of the fit of the frame it estimates, in the abstract and in the limits.
- N5 holds: the part that does not last is called frame-specific (the fit of the frame, or a fluctuation inside the frame that the fast sample sees). N5 fails: reported.
- beta_0 below 1 with its interval below 1 is stated: the corrections are too large by that factor, which is the measured basis of the shrinking second stage.

## 6. Expectations

Floor 126 to 145 eV (root mean square). N1 holds; MSE over the floor 1.25 to 1.45 for the network. Removable-error skill over the causal GP +0.19 to +0.41. N2: 0.60 to 0.85, interval reaching down to about 0.45; may fail. N3: rho_1 0.70 to 0.90, holds. N4: rho_1 0.70 to 0.95, holds. N5: -0.10 to +0.10, holds. beta_0 0.6 to 0.9.

## 7. Limits known in advance

The floor is a lower bound and rests on an assumption about the underlying value that cannot be tested without fit errors; heavy tails make it wide. The persistence test does not see a bias common to consecutive frames. The network's estimates are those of the CPU run of B.55, which differ from the stored ones by 0.026 eV at the median.

## Addendum 1 (2026-11-18, before any run)

Check (ii) compares point estimates (to three significant figures); the intervals of section 2 came from 4,000 two-level draws and this batch uses 2,000. The triples of the floor are consecutive grid rows whose stored time step is below 15 ms at t and at t+1, as in the look.
