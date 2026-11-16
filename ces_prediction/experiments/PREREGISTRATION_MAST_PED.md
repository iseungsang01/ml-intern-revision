# PREREGISTRATION MAST pedestal top — a causal estimate of a hidden charge-exchange frame with co-located Thomson scattering, on the confirmation half

Registered 2026-11-16, before any confirmation shot (even-numbered shots of
`mast/coupling_shots.txt`) is opened for this task. Protocol note of the exploration:
`MAST_FAST_EXPLORATION.md`. Stated aim of the work: nowcasting at the pedestal top, the estimator
and the inputs with the lowest error there against hold and interpolation.

## 1. What the discovery half showed (exploratory; 1,655 odd-numbered shots, 2,301,107 rows)

Loose population (relative error < 0.5, chord good in at least half of the block frames), hidden
5 ms frames, gradient-boosted trees with shot-level 5-fold prediction. Pedestal top = Thomson
T_e / n_e available at the hidden and the last visible frame, chord 1.5 to 5 cm inside the edge of
the Thomson density profile (73,051 rows, 1,439 shots). T_i there: own past vs hold +0.100
[+0.078, +0.118]; Thomson added +0.033 [+0.022, +0.046]; D-alpha added +0.022 [+0.012, +0.031];
all inputs vs hold +0.135 [+0.111, +0.162]; offline interpolation vs hold +0.286; median
absolute error of the all-input model 0.809 of that of hold. More than 20 cm inside: Thomson
added +0.162 [+0.143, +0.180], all inputs vs hold +0.301. The burst detector frozen for the KSTAR
BES stream labels 0.868 of the MAST rows burst-active when applied to D-alpha, so it does not
separate phases on MAST and is not used below.

## 2. Frozen for the confirmation

- Rows, labels and inputs: `mast/explore_ped.py` as committed (86f0490): `shot_rows` with
  population W and share 0.5, `add_extra`, `group_features`.
- Models: `HistGradientBoostingRegressor` (max_iter 300, learning_rate 0.06, max_leaf_nodes 31,
  min_samples_leaf 200, l2 1.0, random_state 20261118), target = hidden value minus last visible
  value, clipped at 20 median reported errors of the discovery rows; trained once on all discovery
  rows; input groups `gb_ces` (own past), `gb_ts` (+ Thomson), `gb_da` (+ D-alpha), `gb_all`
  (+ Thomson, D-alpha, line density, saddle-coil signal and H-mode rating), `gb_all_no_ts`.
- Cells: pedestal top as in section 1; core = more than 20 cm inside the density edge with
  Thomson available; pedestal top with H-mode rating at least 0.5.
- Statistics: 1 - SSE_a / SSE_b on the confirmation rows, 95 % interval from 2,000 shot-level
  bootstrap resamples (seed 20261116). Median absolute error ratios from 400 shot-level resamples.

## 3. Registered statements (T_i; V_i is reported with the same cells and decides nothing)

- **P1.** Pedestal top: all inputs vs hold has an interval above 0.
- **P2.** Pedestal top: Thomson added to the own past (`gb_ts` vs `gb_ces`) has an interval
  above 0.
- **P3.** Pedestal top: D-alpha added to the own past (`gb_da` vs `gb_ces`) has an interval
  above 0.
- **P4.** The Thomson increment is larger in the core than at the pedestal top: the interval of
  the difference of the two skills lies above 0.
- **P5.** Pedestal top: the median absolute error of the all-input model is below that of hold
  (interval of the ratio below 1).
- **P6.** Pedestal top: the all-input model does not reach offline interpolation (`gb_all` vs
  the mean of the two neighbouring visible frames has an interval below 0).

Rules for the paper. P1 and P2 hold: the paper states that on a second device, at the pedestal
top located from the Thomson profile, a causal estimate with a co-located measurement has a lower
error than hold and than the same estimate without it, with the sizes found. P2 fails: the MAST
paragraph says the co-located increment is not confirmed at the pedestal top. P4 and P6 are
stated as found in either direction. Nothing from MAST is written as support for the KSTAR
numbers; MAST carries the statement that the direction of the effect is not specific to one
device and one pair of diagnostics.

## 4. Numerical expectations (stated so they can fail)

- Confirmation rows about 2.2 to 2.4 million; pedestal-top rows 60,000 to 85,000.
- P1: +0.09 to +0.18, holds (probability 0.95). P2: +0.015 to +0.06, holds (0.9). P3: +0.005 to
  +0.04, holds (0.75). P4: core +0.12 to +0.20 against pedestal top +0.015 to +0.06, holds
  (0.95). P5: ratio 0.78 to 0.86, holds (0.95). P6: -0.35 to -0.10, holds (0.9).
- V_i at the pedestal top: all inputs vs hold +0.06 to +0.20; Thomson added +0.02 to +0.10.

## 5. Disclosures

- Known beforehand about the confirmation shots: the lagged correlations of
  `docs/paper/mast_coupling.md` and the regime numbers of `docs/paper/mast_edge.md`. No hidden-frame
  estimate with Thomson, D-alpha or the other inputs has been computed on them.
- The pedestal-top band, the edge definition and every model choice were made on the discovery
  half. Gaps are frames hidden by us; the Thomson system has one to two samples per frame and is
  not faster than the charge-exchange system, only independent of it.
- The trees are a flexible reference, not a real-time form; a small form is not part of this
  registration.
- Output: `docs/paper/mast_ped_confirmation.json` / `.md`; record in `THESIS_RESULTS.md`
  section 8; code `ces_prediction/experiments/mast/confirm_ped.py`. CPU only.
