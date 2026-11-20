# B.78 — Addendum 1 (implementing agent): implementation choices

Written 2026-11-20 before any run of this batch's code and before any readout. No statement, MSE, skill or other score under any rate rule had been computed when this text was written. What had been looked at: the registration; B.75's registration, Addendum 1, code, `docs/paper/b75_availability.md` and `docs/paper/b75_checks.md` (their numbers are known, as the registration says); the B.73 code; and these facts of the inputs (no target, no estimate): the distance of the D1 row stamps from the 10 ms lattice counted in the two ways of section 2, the counts of measured grid rows and of D1 rows, the number of scored rows and their time steps. They are given in section 1.

Facts and choices only. No registered definition is changed. Code: `ces_prediction/experiments/b78_sparse/`; caches `data/.b78/`; outputs `docs/paper/b78_sparse.{md,json}`, `docs/paper/b78_checks.md`. A change made after this text was first written is added in a dated section at the end, with what had been seen at that time.

## 1. Facts looked at before any run

- **Lattice.** Counted from the first grid row of the block, a file's first row (t = 0) included, 1,510 D1 rows lie 2.0 ms from a multiple of 10 ms: in 8 files the first row joins the first block (B.75 Addendum 1 section 1), and the step from it to the next row is not a multiple of 10 ms. Counted from the first row of the block that is not a file's first row, every grid row other than a file's first row lies within 0.001 ms of a multiple of 10 ms.
- **Measured values.** 225,793 grid rows carry a T_i value (cut, held-free grid of `data/.b73/grid.npz`); 222,672 of them are D1 rows; 3,121 are not (2 of them in the six files without D1 rows). As in B.75, every measured value can be a visible value; only D1 rows are scored.
- **Scored rows** (frame index as in section 2, odd and not a multiple of 5): 88,749 D1 rows (0.399 of them). Time step to the previous grid row: 88,391 at 10 ms, 313 at 20 ms, 45 at 30 ms or more.

## 2. Frame index

- f(i) = round((s(i) − s0) / 10 ms), s0 = the stamp of the first row of the block of i that is not a file's first row. Blocks as in B.73 (cut at a step of 0.5 s or more).
- A file's first row (t = 0) and last row (t = 30 s) are not grid rows (`docs/paper/file_rules.md`) and get no index: never kept, never scored. Neither carries a T_i value.
- A row whose stamp is more than 3 ms from s0 + 10 ms × f gets no index: never kept, never scored. None exists (section 1); check 1 counts them.
- The index counts 10 ms steps, not rows: an absent frame (a step of 20 to 40 ms, the beam-off frame not in the file) leaves its index unoccupied.

## 3. Visibility under Kk and the inputs

- **Visible values** for a scored row t: the measured values of the rows of the block of t that lie before t and whose frame index is a multiple of k. Because every measured row has an index (section 1), K1 makes every earlier measured value of the block visible, as rule D10 of B.75.
- **A kept index without a value contributes nothing.** An index that is a multiple of k with no grid row (absent frame), or whose row carries no value (missing, held repeat withheld, above 3 keV), adds no value, and no other value takes its place. The newest visible value is then older than one kept spacing.
- **Inputs**, as B.75 Addendum 1 section 3 with "visible" read as above, in the same operations and order (so that K1 gives D10 bit for bit): H = the newest visible value; age = s(t) minus its stamp, in ms; the anchor row is its row. The five history scalars of the last up to ten visible values (mean, standard deviation with ddof 0, last minus mean, last minus second last whatever their time apart, number kept), undefined with fewer than three. M5 = the median of the last up to five of them. G = `baselines_interpolation.predict_gp_causal` on the last up to 16 visible values, evaluated at s(t). Time step of row t from `rows.npz`. The ten history scalars in B.73's order. The eight fast scalars of row t per unit (`stage_trees.Feats.window` of the unit's fit discharges). The six anchor scalars of `common75.anchor_scalars` (a1, a2, a3 of BES, then of ECEI).
- Inputs are computed for the scored rows only; the other D1 rows stay undefined in `data/.b78/inputs/{rule}.npz` (arrays over the D1 rows in `rows.npz` order).

## 4. Rows of each fit and score

- **Complete under a rule.** The ten history scalars, G, the eight fast scalars of row t and the six anchor scalars are finite (as B.75; finiteness does not depend on the unit).
- **Common set (per target X in F, L5).** Scored rows complete under each of K1, K2, K5, K10, K20, K50, with X defined. Every fit and every score uses it: the fit rows of a unit are the rows of its fit discharges in the set, the test rows those of its test discharges.
- **Counts reported.** Per rule: scored rows with a visible value, of them with fewer than three, rows complete, rows dropped by a3 alone; the common set per target; rows, discharges and clusters scored per design; fit rows per unit.

## 5. Estimators and fits

- Boosted regression `HistGradientBoostingRegressor(random_state=20261116)`, default settings, fitted to (X − G under the rule); the estimate is G under the rule plus its prediction.
- **T0**: the ten history scalars. **TX**: those, the eight fast scalars of row t, the six anchor scalars. **TB** (**TE**): the ten, the four BES (ECEI) fast scalars, a1-a3 of BES (ECEI).
- **Plan.** T0 and TX: all 36 units of B.73, six rules, F and L5. TB and TE: the V1 and V4 units, rules K1 and K10, F and L5.
- **Controls** (V1 units, K10, L5; same fit rows and test rows as TX):
  - **TXp (placebo).** TX's inputs with the fourteen fast and anchor columns permuted jointly over rows, within the fit rows and within the test rows separately (one permutation of each per unit, stream `[20261178, crc32("placebo|<unit>")]`); the ten history columns stay in place. This is the session lead's control of B.75.
  - **TXo (fast scalars at the anchor row).** The ten history scalars, the eight fast scalars of the anchor row, and a1 of BES and of ECEI. With the anchor in place of row t, a2 = 0 and a3 = 0; they are left out (as in the session lead's control of B.75). Rows whose anchor-row fast scalars are not finite are kept, so the rows are those of T0 and TX, and are left to the regression's native handling of missing values; their number is reported.
- H, M5, G are not fitted. **O** = the reported value of row t, scored against L5 only.

## 6. Readouts and intervals

- **Skill** of A over B = 1 − SSE(A) / SSE(B) on identical rows; increment = TX (TB, TE) over T0 under the same rule.
- **Points.** V1: mean of the three re-partitions' values; V2: of the three assignments; V3: fifths 2-5 pooled; V4a, V4b: the test block. RMSE: mean over splits of the per-split RMSE. Ratios and shares: mean over splits of the per-split value.
- **Cost of the lower rate**: SSE(rule) / SSE(K1) for T0 and for TX. **Recovered share**: (SSE(T0) − SSE(TX)) / (SSE(T0) − SSE(T0 under K1)) under the same rule. Neither is formed under K1.
- **Equivalent rate.** For each k in 1, 2, 5, 10, 20, 50: the curve is piecewise linear in ln k through (ln k_j, SSE of T0 under K k_j), j over the six rules, on the same rows. k' = exp of the first point at which the curve, followed from K1 upward, reaches SSE(TX under Kk). If SSE(TX under Kk) is at most SSE(T0 under K1), k' is "below 1" (outside the rules; factor "above k"); if the curve stays below it up to K50, k' is "above 50" (factor "below k/50"). Whether the T0 curve is monotone is reported (at the point; share of draws); if it is not, the first crossing is taken. Point: from the SSEs averaged over the splits (the rows are the same in every split, so this is the MSE averaged over the splits). Interval: k' formed draw by draw on the draws of the cell; the 2.5 and 97.5 % quantiles of ln k' with a censored draw set to ±1000, so that a censored end reads "below 1" or "above 50"; the share of censored draws is reported. Factor k / k' with the ends swapped.
- **Position between kept values** (a stratum on an input). For a scored row under Kk, k ≥ 2: uf = f(t) − f(anchor row), in frames. Thirds of the kept spacing: 3 uf < k; k ≤ 3 uf < 2k; 2k ≤ 3 uf < 3k; and a fourth stratum, uf ≥ k (the newest kept index before t carries no visible value), reported beside the thirds. Integer arithmetic; the same as the age in ms divided by 10 k ms, since the stamps lie within 0.001 ms of the lattice and no boundary falls on a whole frame. By the scored-row rule some thirds are empty by construction: under K2 a row whose newest kept index carries a value has uf = 1 (the middle third); under K10 uf is 1, 3, 7 or 9 (first and last thirds; the middle third is empty). Empty strata are reported as empty. Computed for V1, V4a and V4b, rules K2-K50, F and L5: counts, RMSE of T0 and TX, TX over T0.
- **Beside the registered readouts** (no statement uses them): TX under each rule over T0 under the next denser rule (the form of K-5 for every k); T0 and TX over O (L5); the age of the newest visible value per rule on the common set (quantiles, shares of the position strata).
- **Intervals.** As B.75 Addendum 1 section 6, with seed 20261178: 20,000 draws, 2.5 and 97.5 % quantiles; units (discharges, or shot clusters, with at least one row in the cell) resampled with replacement and the per-unit sums of squared errors re-added. V1: a re-partition drawn first; V2: an assignment drawn first; V3: the units of fifths 2-5 pooled; V4a, V4b: the units of the test block. One stream per cell, `[20261178, crc32("design|target|common|stratum|form")]`, e.g. "V1|L5|common|all|cluster", "V1|L5|common|K10|third 1|discharge". Every quantity of a cell is formed draw by draw on the same draws; this includes K-5 and K-6, which compare different rules on the same rows, and the equivalent rate.

## 7. Statements: how each is judged (cluster form, target L5, common set)

| Statement | Quantity | Holds if |
|---|---|---|
| K-1, K-2, K-3, K-4 | TX over T0 under K10 (V1; V2; V3 pooled over fifths 2-5; V4a) | lower end above zero |
| K-5 | 1 − SSE(TX under K10) / SSE(T0 under K5), V1 | lower end above zero |
| K-6 | SSE(TX under K10) / SSE(TX under K1), V1 | upper end below 2 |
| K-7 | TE over T0 under K10, V1 | lower end above zero |
| K-8 | TXp over T0 under K10, V1 | lower end at most zero |

"V4" without a qualifier is V4a (fit to 31991), as K-4 names it; V4b is given in the note.

## 8. Checks (registration section 10), run before any readout

| Check | What is compared | Pass rule |
|---|---|---|
| frames | the frame index recomputed; rows off the lattice; measured rows without an index; scored rows whose index is a multiple of k, for k = 2, 5, 10, 20, 50 | identical; 0; 0; 0 for every k |
| k1 | H, age, M5, G, the five history scalars, the anchor row and the number of visible values under K1 against `data/.b75/inputs/D10.npz`, on the scored rows | bit for bit |
| visibility | 200 test rows of unit V1_r2_k0 in the common set (L5) under K10 (stream `crc32("visibility|K10")`). For each row a copy of its file in which every T_i value off the kept grid (index not a multiple of 10, or no index) or not before row t is (a) removed, (b) raised by 1000 eV; the ten history scalars, G, M5, the number of visible values, the anchor row and the six anchor scalars recomputed | every row identical bit for bit |
| units | the test discharges of every B.73 unit against the test rows stored in `data/.b73/trees/{unit}__T0__F.npz` | identical |

The eight fast scalars and the running means m(i) take no T_i argument and are not recomputed in the visibility check. The second item of registration section 10 (T0 and TX fitted under K1 on B.75's D10 rows) is not required to match and is not run. No check's tolerance is changed after a run.

## 9. Notes after the checks, before any fit and before any readout (2026-11-20)

- The four checks were run once each and pass (`docs/paper/b78_checks.md`): no row off the lattice, no measured row without an index, no scored row on a kept grid; under K1 every input (H, age, M5, G, the five history scalars, the anchor row, the number of visible values) is bit-identical to B.75's D10 inputs on the 88,749 scored rows; the visibility check is identical on 200 of 200 rows under both copies; every unit has B.73's test rows. Nothing was changed because of them.
- What had been seen at that time besides the checks: the row counts of `data/.b78/sets.json` (common set F 62,162, L5 60,523 rows). No fit had been run; no MSE, skill or statement had been computed.

## 10. After the fits, during the readout (2026-11-20)

- All 1,030 fits (T0 and TX 864, TB and TE 136, the two controls 30) were run once; no fit was repeated or changed. The fit logs carry unit names, fit counts and times only.
- The first run of `readout78.py` stopped with an `IndexError` while it formed the position strata of the first group (V1, target F), before any squared error, MSE, skill or statement had been computed: the anchor rows were taken over all D1 rows instead of the group's rows. The function was changed to select the group's rows (`position_classes`), with no change of any definition, row, fit or statement rule, and the readout was run again from the start. Nothing had been read from the first run but the traceback (`data/.b78/logs/readout_crash1.log`).
- The second run computed every readout in one pass, wrote `docs/paper/b78_sparse.json` and rendered `b78_sparse.md` from it (58 s). It is the only run that computed any score. After it, three points were recomputed from the fit files with separate code (K-1 +0.1908, K-6 1.4301, recovered share of V1, K10, L5 0.5077) and agree with the JSON. No definition, row, fit or statement rule was changed after any number had been read.
