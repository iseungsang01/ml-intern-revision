# PREREGISTRATION B.30 — thinned-CES replay: frozen estimators when a CES sample is visible only every k-th frame

Registered 2026-11-16, before any thinned prediction is computed. Committed before the runner.
Stated aim of the work (2026-11-16): nowcasting at the pedestal-top chord, the best estimator
against interpolation and hold, inside the 10 ms frame, and a demonstration of what it does when
applied. This batch is the demonstration on KSTAR: CES samples that exist are withheld from the
estimators, the 10 ms grid is filled, and the withheld measurements are the answer.

## 1. Why

In the paper 97.4 % of the scored T_i rows have the previous CES sample 10 ms earlier, so the
evaluation is a one-step nowcast and the natural longer gaps (2.6 % of rows) are also the rows
where the fit failed around them. Thinning removes that confound: every row is scored at every
age of the last visible sample.

## 2. Frozen inputs (no training, CPU inference only)

- Data treatment: held (bit-identical repeated) targets removed (`CES_DROP_STUCK_TARGETS=1`);
  population `cut` (`CES_TI_SPIKE_CUT_EV=3000`, primary) and `incl` (`0`, co-reported).
- Splits: the four file-level splits (seeds 42, 1, 7, 123); TEST files of the split manifests
  the paper uses (`run_b29.CONTROLS`).
- Checkpoints, read-only:
  - `bb`: the paper's network, per-discharge standardization (`.b1_seqv2_s{s}_i{s}` cut,
    `.b5i_seqv2_s{s}` incl).
  - `bb80` / `hist80` (cut only): the network and its history-only twin, both retrained under the
    80-epoch cap (`.b29a1_v2_cut_s{s}`, `.b29a1_v2h_cut_s{s}`); the single-variable pair.
  - `hist30` (incl only): history-only twin under the 30-epoch cap (`.b29_v2h_incl_s{s}`), paired
    with `bb`.
  - `causal`: train-file-standardized network (`.b29_norm0_{pop}_s{s}`).
- Each network is run with its own stored normalization statistics and its own standardization
  mode; `CES_SEQ_NORM_EXPANDING=0`, `CES_SEQ_ZERO_MC=0`, no extra channels.

## 3. Thinning

Inside each contiguous block (the block rule of `seq_data.build_blocks`), a row with index i is
visible when (i - o) mod k = 0. At every other row both CES targets are withheld from every
estimator. k in {1, 2, 3, 5, 10} (a visible sample every 10, 20, 30, 50, 100 ms); every phase
o = 0 .. k-1 is run, so each row is predicted once at each age 1 .. k of the last visible frame.
k = 1 is the paper's causal input. The fast channels are untouched at every row.

Check before any thinned number is read: at k = 1 the squared errors of `bb`, persistence and the
causal GP on the stored scored rows of each TEST discharge equal the stored ones
(`comparison_errors_test.npz`; relative deviation <= 1e-3 on >= 99 % of the rows of every split).
If the check fails the batch stops.

## 4. Estimators on the same visible samples

| estimator | reads | note |
|---|---|---|
| hold | last visible sample | persistence |
| causal GP | 16 nearest earlier visible samples | `predict_gp_causal`; the length-scale grid is multiplied by k so that it spans the same range in units of the sampling interval |
| offline PCHIP | visible samples on both sides | not causal; reference only |
| `hist80` / `hist30` | visible CES past | network without fast channels |
| `bb`, `bb80` | visible CES past + fast channels up to the row | whole-discharge standardization |
| `causal` | the same | train-file standardization |

## 5. Scored rows and statistics

- A row is scored for a target when the target is observed there (after the population rule), at
  least one earlier and one later visible observed sample of that target exist in the block, and
  every estimator returns a finite value. One common row set per split, population, target, k and
  phase. The row itself is never an input of its own estimate.
- Per split, population, target and k, pooled over phases: RMSE of each estimator and paired
  skill 1 - SSE_a / SSE_b, 95 % interval from 2,000 discharge-level bootstrap resamples (seed
  20261116). Target T_i is primary; V_rot is reported with the same rules and decides nothing.
- By age under k = 10 (10 .. 100 ms): the same readouts per age bin.
- Co-reported, not deciding: the same skills with the rows in the top 1 % of the hold squared
  error (per split, population, target, k) removed from every estimator; median absolute error.

## 6. Registered statements (T_i, cut population, four splits)

- **H30-1 (the fast-channel share does not vanish and grows with the gap).** `bb80` vs `hist80`:
  the interval lies above 0 at k = 5 on at least 3 of 4 splits, and the point estimate at k = 5 is
  larger than at k = 1 on at least 3 of 4 splits.
- **H30-2 (the network keeps its distance to hold).** `bb` vs hold at k = 5 is not below its
  k = 1 value by more than 0.05 on at least 3 of 4 splits.
- **H30-3 (against the causal GP at a sparse cadence).** `bb` vs causal GP: interval above 0 at
  k = 5 on at least 3 of 4 splits.
- **H30-4 (causal variant).** `causal` vs causal GP: interval above 0 at k = 5 on at least 3 of 4
  splits.
- **H30-5 (a causal estimate with fast channels against offline interpolation of sparse CES).**
  `bb` vs offline PCHIP: interval above 0 at k = 5 on at least 3 of 4 splits.

Rules for the paper. H30-1 holds: the paper states that the fast-channel share grows when CES is
sparser, with the table. H30-1 fails by the first clause: the paper states that the fast-channel
share measured at 10 ms does not carry over to sparser CES for the frozen network. H30-2 or H30-3
fails: the thinned replay is reported as a limit of the frozen network (trained at the natural
cadence), and retraining with imposed dropout is named as the next step. No headline number of
the paper is replaced by a B.30 number; the replay is a separate section.

## 7. Numerical expectations (stated so they can fail)

From the noisy-OU fit of the record (robust T_i: noise 39 eV, signal 69 eV, 81 ms): the hold MSE
grows by about 1.8 at 50 ms and 2.4 at 100 ms; a CES-only filter keeps about 0.35 to 0.40 of
skill over hold at every lag; the innovation share of its error rises from 0.41 (10 ms) to 0.69
(50 ms).

- `bb` vs hold: k = 1 +0.30 to +0.40; k = 5 +0.30 to +0.50; k = 10 +0.25 to +0.50.
- `bb80` vs `hist80`: k = 1 +0.06 to +0.12 (known); k = 5 +0.08 to +0.20; k = 10 +0.08 to +0.22.
- `bb` vs causal GP: k = 5 +0.10 to +0.30.
- `bb` vs offline PCHIP: k = 1 +0.05 to +0.20 (known sign); k = 5 +0.05 to +0.25.
- Probability that the frozen recurrent state misbehaves at staleness it rarely saw (H30-2
  fails): about 0.3.
- Hold RMSE (cut, with tails): 180 to 200 eV at k = 1, 220 to 280 eV at k = 5.

## 8. Disclosures

- The networks were trained at the natural cadence; inputs at k >= 3 are outside most of their
  training distribution. Nothing is retrained here.
- Withheld samples are measured CES values; a thinned cadence is imposed, not a property of the
  diagnostic.
- `bb`, `bb80`, `hist80`, `hist30` standardize the fast channels with whole-discharge statistics
  (not causal); `causal` does not. The standardizer never reads CES.
- The GP grid scaling is a choice made here; the unscaled grid is reported beside it for k = 5.
- The row population is the grid population defined in section 5, not the window population of
  the paper's harness; k = 1 numbers are therefore close to, not identical with, the paper's.
- Output: `docs/paper/b30_thinned.json` / `.md`, record in `THESIS_RESULTS.md` section 8.
  Runner: `ces_prediction/experiments/b30_thinned/run_b30.py`.
