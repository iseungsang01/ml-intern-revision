# B.44 — in which discharges the margins hold: strata of the test discharges

Registered 2026-11-18, before any of the readouts below was computed. CPU only; nothing is trained. Designed by the session lead and executed by a delegated agent from this document.

## 1. Why

The tenth independent grading: the paper does not say under which conditions its result holds, because the files carry no confinement-mode label and no equilibrium. Strata that the files do allow can be formed per discharge, and the two margins of the paper (the causal variant over the history-only twin, and over the causal GP) read inside each.

## 2. Seen before

The per-discharge relation between the regime number and the ordering of the estimators (manuscript). B.33: in the cluster of rows with a low BES level the fast-channel increment is −0.016 to +0.049. B.41: the increment inside rows that pass the value rules (+0.044 to +0.152) and the share of the margin over the GP on flagged rows. The split by time (later discharges; the causal variant +0.103 to +0.112 over the GP). Not seen: either margin by a stratum of discharges.

## 3. Design

Frozen B.30 row tables `data/.b30/rows_cut_s{42,1,7,123}.npz`, T_i, cut population, natural cadence, rows with `hold`, `gp`, `hist80`, `causal` and the measurement finite. Strata of the test discharges of a split, each a division into thirds by a quantity computed from the 10 ms file of the discharge (grid rows, held values and values above 3 keV removed):

1. the median measured T_i;
2. the robust white component over the median T_i (B.40: 1.4826 × the median absolute deviation of the second difference over √6, divided by the median T_i);
3. the share of rows in which at least one BES channel departs by more than +3 standard deviations from its 100 ms mean (B.32 definition);
4. the number of absent frames (beam blips) per second of measured time;
5. the shot number (the order in the campaign).

Inside each third of each stratum: the skill of the causal variant over the twin, over the causal GP and over hold, with 95 % intervals (2,000 resamples of the discharges of that third). A statement holds if it holds on at least three of the four splits.

## 4. Statements

- **W1.** In every third of the median T_i the causal variant has skill over the history-only twin with the interval above zero.
- **W2.** In every third of the median T_i the causal variant has skill over the causal GP with the interval above zero.
- Reported: the same tables for the other four strata; for each stratum the difference of the skill between the highest and the lowest third with its interval; the number of discharges and rows in each cell; the thirds' edges.

## 5. What goes into the paper

- W1 and W2 hold: the paper says that both margins are present at low, middle and high temperature of the chord. Either fails: the paper names the third in which the margin is not established and bounds the claim to the others.
- A stratum whose highest-minus-lowest difference has an interval excluding zero on at least three splits is reported as a condition on which the margin depends.

## 6. Expectations

W1 does not hold: in the lowest third of the median T_i the increment is not separated from zero on two or more splits (+0.00 to +0.06), against +0.05 to +0.15 in the highest. W2 holds on two or three splits. The increment grows with the share of rows with a large upward BES departure and does not depend on the shot number.

## 7. Limits known in advance

About 32 discharges per third of a split, so the intervals are wide. The strata are computed from the same files as the scores. They are not confinement modes.
