# PREREGISTRATION regime transfer — does the MAST relation between the regime number and the right CES-only estimator hold across KSTAR discharges

Registered 2026-11-15, before any per-discharge skill on KSTAR is related to the regime number.
Committed before the analysis code. CPU only, real data and frozen artifacts only, no neural
network is trained.

## 1. Why

On MAST (88 public TEST discharges, registered gate, `docs/paper/mast_gate.json`) the per-discharge
regime number (lag-1 autocorrelation of 10 ms increments) is related to which CES-only estimator
is the right bar: the Spearman correlation between the regime number and (slope-filter skill -
running-mean skill, both against persistence) is +0.523 [+0.326, +0.682] for T_i and
+0.759 [+0.624, +0.856] for rotation. KSTAR as a whole is noise-limited (median regime number
-0.416 [-0.429, -0.402] for T_i, -0.349 [-0.384, -0.322] for rotation), but the discharges are
spread (T_i quartiles in `docs/paper/ppcf/kstar_regime.json`). A statement confirmed on one device
is carried to the other as a prediction and tested on data already held.

## 2. State of knowledge at registration

- Seen: the distribution of the KSTAR regime number over discharges and its median
  (`kstar_regime.json`); every pooled KSTAR skill (backbone, causal GP, running mean,
  persistence, PCHIP) per split and over the repartitions; the MAST relations above.
- Not seen: any KSTAR skill per discharge, any relation between a KSTAR per-discharge skill and
  the regime number, any split of KSTAR skill by regime number.

## 3. Data and definitions (fixed)

- Rows: the frozen TEST rows of the eight canonical cells (four file-level splits, seeds
  42 / 1 / 7 / 123, populations cut and inclusive), read through
  `review_loop/analyses/R2/r2_common.canonical` (backbone, persistence, causal GP, PCHIP squared
  errors on identical rows; row index from the stage-4 file after the bit-identity check).
- Running mean (CES-only, causal): exponentially weighted mean of the past observed held-free
  values of the same contiguous segment (gap < 0.5 s), update m <- m + alpha (y - m), first
  observation taken whole, prediction = state before the row; alpha grid {0.05, ..., 1.00};
  alpha chosen on the VAL files of the same split by SSE (rows with a past observation).
  T_i: every observation enters. Rotation: the gated arm and the 1000 km/s row rule of the
  round-2 analysis, unchanged (`r2ac_vrot_running_mean.py`; the rule was set post hoc there).
- Regime number per discharge: `docs/paper/ppcf/kstar_regime.py`, unchanged, from all CES rows of
  the discharge (T_i above 3 keV treated as missing in both populations; rotation holds and
  values beyond 1000 km/s treated as missing; at least 8 increment pairs). One number per
  discharge and target, the same in both populations.
- Per-discharge sums: a discharge can be in the TEST set of more than one split. Its squared-error
  sums are added over every split in which it appears (each arm summed over the same rows).
- Included discharge: regime number defined; mean number of scored rows per split in which it
  appears >= 30; persistence and causal-GP sums > 0.
- Per-discharge quantities:
  - D1 = (SSE_running-mean - SSE_causal-GP) / SSE_persistence, i.e. causal-GP skill minus
    running-mean skill, both against persistence.
  - D1b = 1 - SSE_running-mean / SSE_persistence (running-mean skill against persistence).
  - D2 = 1 - SSE_backbone / SSE_causal-GP.
- Statistic: Spearman rank correlation across discharges between the regime number and each
  quantity; 95 % percentile interval from 2,000 bootstrap resamples of discharges, seed 20261116.
  Reported per target (T_i primary, rotation co-reported) and population (cut primary,
  inclusive co-reported).
- Co-reported, not deciding: discharges split into tertiles of the regime number; pooled
  (ratio-of-sums) skill per tertile for causal GP vs running mean, running mean vs persistence,
  backbone vs causal GP, with discharge-cluster bootstrap intervals (2,000, same seed).
- Row alignment check before any statistic: per split, the pooled backbone-vs-causal-GP skill and
  (rotation) the running-mean readouts must reproduce the stored round-2 values; the persistence
  anchor recomputed from the row index must match the stored persistence squared error on every
  row.

## 4. Hypotheses and rules

- **T1 (transfer of the MAST relation).** Across KSTAR discharges, a higher (less negative) regime
  number goes with a larger advantage of the causal GP over the running mean: Spearman(regime
  number, D1) > 0. T1 holds for a target if the interval excludes 0 above 0 in the cut population.
  Co-reported direction: Spearman(regime number, D1b) < 0 (averaging pays where increments are
  noise-dominated).
- **T2 (link to what the backbone adds).** Prediction, one direction chosen in advance:
  Spearman(regime number, D2) > 0 for T_i. Reason: the margin of the backbone over the causal GP
  is carried by rows where the plasma moved; a discharge whose increments are less noise-dominated
  has more such structure for the co-located fast channels to follow, while in a fully
  noise-dominated discharge the CES-only GP is already near what any estimator can do. The opposite
  reading (the noisier the CES series, the more an independent fast measurement helps) is the
  alternative; if the interval excludes 0 below 0, that reading is reported. T2 holds if the
  interval excludes 0 above 0 in the cut population.
- Writing rule. T1 holds for T_i: the paper states that the regime number orders the CES-only
  estimators inside KSTAR as well, with the correlation and the tertile skills. T1 does not hold:
  the paper states that the relation found on MAST is not detected in the discharge-to-discharge
  variation on KSTAR, with the interval. T2 is reported with its interval in either case and is
  not written as evidence of the fast-channel share (no history-only backbone is involved).

## 5. Numerical expectations (stated so they can fail)

- Included discharges: T_i 250-450 (cut), rotation 100-250.
- T1, T_i, cut: Spearman +0.15 to +0.45, interval excluding 0. D1b: -0.2 to -0.5.
- T1, rotation, cut: +0.05 to +0.35; the interval may include 0 (fewer discharges, fit noise).
- T2, T_i, cut: +0.05 to +0.25; the interval may include 0.
- Tertiles, T_i, cut: running-mean skill against persistence falls from the lowest to the highest
  regime tertile; backbone vs causal GP rises.

## 6. Disclosures

- The regime number is computed from the same discharge whose TEST rows are scored (as on MAST,
  where it came from the visible frames of the scored discharge). The relation is within-discharge
  description, not out-of-sample prediction of a discharge from other discharges.
- A discharge that appears in several splits is scored by different trained backbones; its sums
  are added, not averaged, so discharges with more appearances weigh the same in a rank statistic
  but are less noisy.
- The KSTAR regime numbers span mostly negative values; the MAST relation was measured on mostly
  positive values. A null result here bounds the relation in the noise-limited range only.
- Rotation uses the post hoc 1000 km/s rule.
- Output: `docs/paper/review_loop/analyses/R3/regime_transfer.json`, record in
  `THESIS_RESULTS.md` section 8. The paper is not edited in this step.
