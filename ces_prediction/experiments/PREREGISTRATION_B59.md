# B.59 — what the estimator must read, with pooled power: one diagnostic at a time, no Mirnov, and fast channels one frame old in training

Registered 2026-11-18, before any run of this batch. GPU, 60 runs; designed by the session lead, runner to be written and checked without training by a delegated agent. Nothing in the shared training code is edited.

## 1. Why

Three statements of the manuscript rest on four splits with one initialisation or on networks that were not retrained, and the pooled readout has shown what 96 discharges cannot decide.

(1) Which diagnostic carries the increment. On the four splits the BES-only network has the increment (4 of 4) and the ECEI-only network does not (1 of 4); ECEI is the diagnostic that does not depend on the beam, and one initialisation on 96 discharges has little power.

(2) The Mirnov channels. With the Mirnov inputs of the trained estimator held at their mean the increment is +0.053 [+0.028, +0.078] (B.55); a network trained without them was not separated from the estimator on any of four splits (B.39). A network given a mean is not a network trained without the channels.

(3) Present frame or earlier frames. With the fast channels one frame old the trained estimator loses the increment (-0.027 [-0.050, -0.008] against its twin, B.55). That estimator was trained with the fast channels of the present frame, so the readout does not say how much a network that only ever sees earlier fast channels can learn. The author's question (2026-11-18): can the fast channels be predicted from earlier data, and is the estimator then weaker.

## 2. Seen before

B.39: pooled, the estimator over the twin +0.048 [+0.022, +0.073]; four splits, one initialisation: BES-only over the twin +0.026 to +0.105 (4/4), ECEI-only +0.001 to +0.084 (1/4), no-Mirnov over the twin +0.043 to +0.135 (3/4), the estimator over the no-Mirnov network -0.022 to +0.024 (0/4). B.55 (trained estimator, changed inputs, pooled): all fast channels one frame old -0.027 against the twin; BES alone +0.000 [-0.024, +0.022]; ECEI alone +0.029 [+0.006, +0.050]; Mirnov at its mean +0.053. Not seen: any of these networks trained on the pooled folds; any network trained with fast channels one frame old.

## 3. Design

T_i, cut population, the protocol of the pooled arm of B.39 (epoch cap 80 with early stopping, standardization with the statistics of the training discharges, initialisation seed of the fold), on the fifteen folds of re-partitions r2, r3 and r4. The estimator and the twin of these folds exist (B.39). Four arms, fifteen runs each, in this order:

- **L** — the estimator with every fast channel replaced, in training and in scoring, by its value one row earlier in the same block (the first row of a block keeps its own). Nothing else changes.
- **N** — the network without the Mirnov channels (`v2n`).
- **E** — the network that reads ECEI and the CES history (`v2e`).
- **B** — the network that reads BES and the CES history (`v2b`).

Scored by `eval_seq` on the test rows; the estimator, the twin and the references are the stored runs of B.39 on the same folds. Interval: a re-partition is drawn among the three, then its discharges are resampled (20,000 draws).

Checks before any GPU run (the batch does not start if one fails): a dry run of every arm; arm L changes the fast channels and nothing else, in the training data and in the scoring data alike, and with a shift of zero rows its inputs are identical, array by array, to those of the estimator of B.39; the first row of every block keeps its own values; arms N, E and B give the environment of the matching runs of B.39 arm A except for the split, the initialisation seed and the output directory; every run is scored on the rows of the estimator of its fold.

## 4. Statements (three re-partitions pooled)

- **L1.** Trained with the fast channels one frame old, the estimator has skill over the twin: interval above zero.
- **L2.** The estimator of B.39 has skill over that network: interval above zero.
- **N1.** The network without Mirnov has skill over the twin: interval above zero.
- **N2.** The estimator of B.39 and the network without Mirnov are equivalent: the interval of the skill of one over the other lies inside [-0.01, +0.01].
- **E1.** The ECEI network has skill over the twin: interval above zero.
- **B1.** The BES network has skill over the twin: interval above zero.

Reported: every network over the causal GP, hold and the offline GP; the estimator over the ECEI and over the BES network; the BES network over the ECEI network; in the rows 1 to 3 after an absent frame and in the other rows, every network over the twin; the median absolute errors; epochs run; the three re-partitions one by one.

## 5. What goes into the paper

- L2 holds and L1 fails: the paper says that what the fast channels add is of the present frame: a network that sees only earlier fast channels does not have the increment. L1 holds: the paper gives the part of the increment that earlier fast channels carry (L1) beside the part that needs the present frame (L2), and the condition on the acquisition of the fast channels is stated with both numbers.
- N1 and N2 hold: the paper says that the estimator does not need the Mirnov channels, and the estimator without them is named as the one to use; making it the estimator of the main table is a change of design put to the author. N2 fails with the estimator better: the Mirnov channels stay and the result of B.55 is stated as a property of the trained network only.
- E1 holds: the increment does not rest on a beam-dependent signal alone, and the beam caveat is rewritten with the pooled numbers of E1 and B1. E1 fails and B1 holds: the increment is stated as resting on the beam emission, with the pooled interval of the ECEI network beside it.
- Failed statements are reported as failed.

## 6. Expectations

L1: -0.01 to +0.02, does not hold. L2: +0.03 to +0.06, holds. N1: +0.035 to +0.06, holds. N2: -0.008 to +0.012, interval about 0.012 wide on each side; may fail by width. E1: +0.01 to +0.03, holds or fails by a small margin. B1: +0.035 to +0.06, holds.

## 7. Limits known in advance

Three of the eight re-partitions; one training per fold. An ablation by removal shows what a network can do without a group, not what the estimator uses. The folds re-partition the discharges on which the design was adopted. Arm L delays the fast channels by one row, 10 ms where no frame is absent and more across an absent frame.
