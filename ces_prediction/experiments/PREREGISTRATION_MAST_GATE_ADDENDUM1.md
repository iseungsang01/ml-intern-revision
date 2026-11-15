# PREREGISTRATION MAST gate, addendum 1 — wider tuning grids for the filter and the GPs

Registered 2026-11-15, after the first run (registration `PREREGISTRATION_MAST_GATE.md`, commit
9f83937; code f6e6f00; result 4832d59, `docs/paper/mast_gate.json` / `.md`) and before any number
of this addendum is computed. Committed before the addendum code.

## 1. Why

In the first run the TUNE optimum of three hyperparameter families sits on the edge of the grid.
Selected values (`mast_gate.json`, `tuning`; identical for `T_i` and `V_i` in units of the target
scale, scale = median reported error of the TUNE rows = 26.91 eV and 1.098 km/s):

| hyperparameter | first-run grid | selected | position |
|---|---|---|---|
| running-mean tau | 5, 10, 15, 25, 40, 70 ms | 5 ms | lower edge |
| filter q_level / scale^2 | 10^(-2, -1.3, -0.6, 0.1, 0.8, 1.5) | 31.62 | **upper edge** |
| filter q_slope / scale^2 | 10^(-4 ... 1), 6 values | 1 | interior |
| causal GP length | 10, 20, 40, 80, 160 ms | 40 ms | interior |
| causal GP amplitude / scale | 0.5, 1, 2, 4 | 4 | **upper edge** |
| offline GP length | same | 10 ms | **lower edge** |
| offline GP amplitude / scale | same | 4 | **upper edge** |

The TUNE error is still falling at those edges (`T_i`, causal GP at 40 ms: TUNE MSE 18,349 at
amplitude 2 and 9,403 at amplitude 4; filter at q_slope = 1: 5,852 at q_level 6.31 and 5,677 at
31.6). So the slope-filter and GP skills of the first run may be understated, and the "best
CES-only causal arm" read by gate (b) may not be the best arm of its family.

## 2. The single change

Wider tuning grids (section 3). Everything else is the first run, unchanged and reused from
`mast/run_gate.py`: the 300-shot list `mast/gate_shots.txt` and the files read (only those shots),
the TUNE / TEST split, populations G and W, good-sample rule, block, scored chords and scored rows,
the nominal 5 ms grid for every weight, the seven arms and their definitions (including the GP
window, the constant-kernel variance 10^6 x amplitude^2 and the diffuse filter start), the target
scale, tuning criterion (pooled MSE on TUNE scored rows, population G, per target; W uses the
G-tuned values), every readout, the bootstrap (2,000 shot-clustered resamples of TEST shots, seed
20261116, the same indices for every arm) and both gate rules with their thresholds. No neural
network, CPU only. `run_gate.py` is not changed in behaviour; `mast_gate.json` stays as it is.

## 3. New grids (old values kept)

| hyperparameter | addendum grid | new values |
|---|---|---|
| running-mean tau | 2.5, 5, 10, 15, 25, 40, 70 ms | 2.5 ms |
| filter q_level / scale^2 | 10^(-2 + 0.7 k), k = 0 ... 8 | 10^2.2, 10^2.9, 10^3.6 (158, 794, 3981) |
| filter q_slope / scale^2 | 10^(-4 ... 1), 6 values | none |
| GP length (both GPs) | 2.5, 3.54, 5, 7.07, 10, 20, 40, 80, 160 ms | 10 / 2^(k/2) ms, k = 1 ... 4 |
| GP amplitude / scale (both GPs) | 0.5, 1, 2, 4, 8, 16, 32 | 8, 16, 32 |

The two GPs share one grid (as in the first run) and are tuned separately on it. The joint optimum
of each family is the grid point with the lowest pooled TUNE MSE, as before.

## 4. Stopping rule (fixed in advance)

After a tuning round, per target, every hyperparameter whose selected value is the first or last
value of its grid is "on an edge". If any is, that edge is extended by three more steps of the
same log-step and the target is retuned on the union grid; at most two such further rounds (three
tuning rounds in total per target). A hyperparameter still on an edge after the last round is
reported as **edge-bound**, with the relative TUNE-MSE change over its last step.

- Steps: q_level x 10^0.7, q_slope x 10, GP amplitude x 2, GP length / sqrt(2) downward and x 2
  upward, tau upward 110, 180, 300 ms then x 1.6.
- The GP grid is shared: it is extended when either GP is on that edge; edge flags are reported
  per GP.
- Exception, tau lower edge: a running mean with tau -> 0 is persistence, which is already an arm.
  If tau = 2.5 ms is selected it is reported as edge-bound at the persistence limit and the grid is
  not extended downward.
- TEST is evaluated once per target, after the last tuning round, with the values selected then.
  No TEST number enters any tuning or stopping decision.

## 5. Statements registered in advance

1. Gate (b) is re-read with the retuned arms: best causal arm = the causal arm with the lowest TEST
   MSE in population G among persistence, extrapolation, retuned running mean, retuned slope-state
   filter, retuned causal GP; gate (b) passes for a target if the 95 % interval of its mean reduced
   chi-square lies above 1.5 in population G. Same threshold, same rule.
2. The first run stays on record as the registered primary. The addendum is reported next to it,
   number by number; it does not replace it. Where the two differ in a verdict, both verdicts are
   stated.
3. Gate (a) uses no arm and must reproduce the first run exactly; so must persistence,
   extrapolation and offline interpolation. This is the integrity check of the reuse.
4. All other first-run readouts (skill against persistence, mean and median reduced chi-square,
   R1, R2, heavy-tail audit, radius tertiles, irregular-shot sensitivity) are recomputed with the
   retuned arms and reported; none of them gates.

## 6. Numerical expectations (guesses from the first run; stated so they can fail)

All TEST, population G, skill = 1 - SSE_arm / SSE_persistence, point estimate unless an interval is
named. First-run values in parentheses.

- Slope-state filter vs persistence, `T_i`: +0.07 to +0.14 (+0.076 [-0.006, +0.148]).
- Slope-state filter vs persistence, `V_i`: +0.08 to +0.15 (+0.088 [+0.035, +0.147]).
- The `T_i` slope-filter interval excludes 0 (above), lower bound between 0 and +0.04 (first run:
  includes 0). The `V_i` interval excludes 0 (above).
- Causal GP vs persistence: `T_i` -0.05 to +0.12 (-0.683), `V_i` -0.05 to +0.13 (-0.350); for both
  targets the causal-GP point is not more than 0.03 above the slope-filter point.
- Best causal arm: slope-state filter for both targets (as in the first run).
- Gate (b) re-read: passes for both; mean reduced chi-square of the best causal arm 2.2 to 3.3 for
  `T_i` (2.71 [2.07, 3.51]) and 16 to 26 for `V_i` (21.20 [16.83, 26.06]).
- Offline GP vs persistence: `T_i` +0.50 to +0.62 (+0.374), `V_i` +0.60 to +0.70 (+0.577).
- Running mean: tau = 2.5 ms selected for both targets; skill between -0.03 and 0.
- Edges after the stopping rule: q_level interior for both targets; the GP amplitude still
  edge-bound for at least one GP (the error flattens toward the noise-free limit without a
  minimum); offline GP length interior.

## 7. Disclosures

- The TEST results of the first run have been seen (every table of `mast_gate.md`), and so have
  its TUNE error surfaces (`mast_gate.json`, `tuning`), which are the basis of section 6. The grid
  change is therefore post hoc with respect to TEST; this is why the first run stays primary
  (statement 2) and why the widening follows a mechanical rule (edge -> three more steps) rather
  than a choice made on TEST numbers.
- Tuning touches TUNE shots only; TUNE and TEST shots are disjoint.
- Very large GP amplitudes make K + R poorly conditioned (noise-free limit). If a grid point
  yields a non-finite TUNE error it is dropped from the grid and counted.
- Only the 300 gate shots are read. The other FAIR-MAST shots still downloading into
  `data/mast/act/` are not read.
- The remaining disclosures of the first registration (section 7) apply unchanged.
- Output: `docs/paper/mast_gate_addendum1.json` / `.md`, record in `THESIS_RESULTS.md` section 8,
  script `mast/run_gate_addendum1.py`.
