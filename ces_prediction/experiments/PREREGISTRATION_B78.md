# B.78 — a CES value every k-th frame: the level in between, with and without the fast diagnostics

Registered 2026-11-20, before any run of this batch. CPU only, a causal GP and boosted regressions on cached rows; no network. The session lead designed it; the author asked that every candidate proposal be tried ("candidate D"). An AI agent implements and runs it under the author's supervision. No existing file is edited.

## 1. Why

B.75 delayed the CES history. Another way in which a CES value can be scarce is its rate: the value exists only every k-th frame. The proposal to test: **with the fast diagnostics of the present, a CES value taken every k-th frame carries the level in between.** The data allow a test that the related work on other diagnostics could not make: the values of the withheld frames exist, so the reconstruction is scored against real measurements at the full rate.

**Status.** Exploratory on used data. Rows, targets, designs, learners and the fast and anchor inputs are those of B.73 and B.75. The rate rules are new and fixed here before any number under them has been read.

## 2. Data, targets, designs

- Rows: the D1 rows of B.70 as cached by B.73 (`data/.b73/rows.npz`, `grid.npz`).
- Targets of a scored row t: F (the reported value of row t) and L5 (the following level: median of the measured values among rows t+1 to t+5, at least three).
- Designs: the units of B.73 (V1, V2, V3, V4 with fits to shot 31991 and to shot 32310).

## 3. Rate rules

The frame index of a row is the number of 10 ms steps from the first row of its block (rounded).

| Rule | Visible values for a scored row t |
|---|---|
| K1 | every measured value of the block before t (the full rate; rule D10 of B.75) |
| K2, K5, K10, K20, K50 | the measured values of the block before t whose frame index is a multiple of k (a value every 20, 50, 100, 200, 500 ms) |

- **Scored rows:** D1 rows whose frame index is odd and not a multiple of 5. Such a row is on no kept grid, so the same rows are scored under every rule.
- The fast samples of rows up to and including t are visible under every rule.
- The targets are built from all measured values, whatever the rule: the withheld frames are the ground truth.

## 4. Estimators

As B.75, fitted per rule and target: H (newest visible value), M5, G (`predict_gp_causal` on the last up to 16 visible values), T0 (G plus boosted regression on the ten history scalars of the visible values), TX (T0's inputs plus the eight fast scalars of row t and the six anchor scalars, the anchor being the row of the newest visible value), TB and TE (the BES and the ECEI half), O (the frame's own reading, against L5 only). `HistGradientBoostingRegressor(random_state=20261116)`, default settings.

## 5. Rows of each fit and score

- **Common set (per target):** scored rows (section 3) complete under each of the six rules (ten history scalars, G, fourteen fast scalars finite), target defined. Every fit and score uses it.
- T0, TX: every unit, every rule, both targets. TB, TE: V1 and V4, rules K1 and K10.

## 6. Readouts

Per design, target and rule on identical rows: RMSE of H, M5, G, T0, TX and of O against L5; T0 over G and over H; the increments TX, TB, TE over T0; the cost of the lower rate MSE(rule) / MSE(K1) for T0 and for TX; the recovered share (MSE(T0) − MSE(TX)) / (MSE(T0) − MSE(T0 under K1)).

**Equivalent rate.** For each k, the rate k' at which T0 would have the MSE of TX under k, by linear interpolation of T0's MSE in ln k between the rules (reported as k' and as the factor k / k').

**By position between two kept values (a stratum on an input):** the increments by the age of the newest visible value in units of the kept spacing (thirds).

**Controls (V1, K10, L5):** the fourteen fast scalars permuted over rows and TX refitted; the fast scalars taken at the anchor row instead of row t.

**Intervals.** 95 %, 20,000 draws, shot clusters resampled (the discharge form beside it), as B.73. A statement is judged on the cluster interval.

## 7. Statements (target L5; "increment" = TX over T0 under the same rule)

| Id | Statement | Expectation |
|---|---|---|
| K-1 | V1, K10: the increment lies above zero | holds (+0.10 to +0.30) |
| K-2 | V2, K10: the increment lies above zero | holds |
| K-3 | V3, K10: the increment lies above zero | uncertain, leaning to holds |
| K-4 | V4 (fit to 31991), K10: the increment lies above zero | uncertain |
| K-5 | V1: the MSE of TX under K10 lies below that of T0 under K5 (with the fast inputs, a value every 100 ms is not worse than a value every 50 ms without them) | uncertain |
| K-6 | V1: MSE(TX under K10) / MSE(TX under K1) lies below 2 | holds (1.3 to 2.0) |
| K-7 | V1, K10: TE over T0 lies above zero | holds |
| K-8 | V1, K10, placebo: the interval of TX over T0 contains zero or lies below it | holds |

Each statement is judged separately; the tables are reported whole.

## 8. What goes into the paper (decided now)

- The table of RMSE by rate is printed whatever the verdicts.
- If K-1 and K-2 hold: the paper states the proposal for sparse CES with the sizes found; K-3 and K-4 decide whether a forward-in-time design is named. If K-5 holds, the equivalent-rate sentence is used.
- If K-1 fails: the paper says the fast diagnostics do not carry the level between sparse CES values, and candidate D is not proposed.
- If K-8 fails, the batch is not used until the cause is found.

## 9. Limits known in advance

- Exploratory on used data.
- The rate is lowered by us after the fact. The withheld frames were taken with the beam on; an operation that takes fewer CES frames would change the beam schedule and with it the plasma between the frames.
- The kept values are the offline product.
- One estimator per rule; the phase of the kept grid is one (frame index a multiple of k).
- The following level is built from reported values.

## 10. Checks and files

- Under K1 the inputs equal B.75's D10 inputs on the scored rows (bit for bit), and T0 and TX fitted under K1 on B.75's D10 rows are not required to match (the row sets differ).
- Visibility: for 200 random test rows under K10 the inputs are recomputed from a copy of the file in which every T_i value off the kept grid or not before t is (a) removed and (b) raised by 1000 eV; all history inputs and the anchor row are unchanged bit for bit.
- Code: `ces_prediction/experiments/b78_sparse/` (may import `b75_availability/common75.py` and `b73_targets/`; edits nothing). Caches: `data/.b78/`. Output: `docs/paper/b78_sparse.{md,json}`, checks `docs/paper/b78_checks.md`.
- Seed 20261178, one stream per quantity. Implementation choices left open go into `PREREGISTRATION_B78_ADDENDUM1.md`, written before any readout. This file is not edited after the first run.
