# Project Knowledge

This file records prior attempts, known constraints, and directions to avoid so future work does not repeat the same failed paths.

## Current Status

- Primary reference point: best validation loss `0.4834` at iteration 7 of the newer architecture-search round.
- Best-known train loss at that point: `0.4494`.
- Use the iteration 7 design as the baseline to preserve, reproduce, and tune.
- Later runs are mainly evidence about what moved away from the best result, not better starting points.
- The project should preserve the best baseline while still allowing genuinely new architecture exploration that does not duplicate failed paths.

## Data And Model Contract

Every generated or edited `model.py` must preserve this interface and data contract:

- `model.forward` must accept `forward(self, bes, ecei, mc, time_features=None, ces_history=None)`.
- Outputs must be normalized `[CES_TI, CES_VT]` with shape `(batch, 2)`.
- Do not denormalize inside `model.py`; inverse transforms belong in evaluation/reporting code.
- BES, ECEI, MC, and targets use train-file-only per-channel z-score normalization.
- `ces_history` has shape `(batch, window, 3)` containing normalized previous `CES_TI`, normalized previous `CES_VT`, and an observed mask.
- The target timestep CES values must remain masked as `[0, 0, 0]` in `ces_history` to avoid leakage.
- Time features encode irregular sampling and currently have 4 channels: lookback seconds, delta seconds, `log1p` lookback, and `log1p` delta.

## What Has Already Been Tried

Older 8-iteration handoff history:

- Iteration 1: train `0.3377`, val `0.5085`.
- Iteration 2: train `0.3289`, val `0.5149`.
- Iteration 3: train `0.3363`, val `0.5326`.
- Iteration 4: train `0.3371`, val `0.5192`.
- Iteration 5: train `0.4423`, val `0.4901` (best recorded).
- Iteration 6: train `0.9789`, val `0.8764` (unstable failure).
- Iteration 7: train `0.4003`, val `0.4967`.
- Iteration 8: train `0.4170`, val `0.5023`.

Newer 18-iteration architecture-search round:

- Iteration 1: train `0.4310`, val `0.5231`.
- Iteration 2: train `0.9742`, val `0.8886` (unstable failure after aggressive scaling).
- Iteration 3: train `0.9692`, val `0.8860` (continued unstable failure).
- Iteration 4: train `0.4350`, val `0.4981` (major recovery; Pre-LayerNorm was identified as the turning point).
- Iteration 5: train `0.4337`, val `0.4966`.
- Iteration 6: train `0.4519`, val `0.4864`.
- Iteration 7: train `0.4494`, val `0.4834` (best recorded; attention pooling was identified as the likely useful change).
- Iteration 8: train `0.4422`, val `0.4981`.
- Iteration 9: train `0.4356`, val `0.5074`.
- Iteration 10: train `0.4291`, val `0.5014`.
- Iteration 11: train `0.4309`, val `0.4881`.
- Iteration 12: train `0.4280`, val `0.5023`.
- Iteration 13: train `0.4306`, val `0.4935`.
- Iteration 14: train `0.4154`, val `0.5057`.
- Iteration 15: train `0.4418`, val `0.5003`.
- Iteration 16: train `0.4212`, val `0.5150`.
- Iteration 17: train `0.4128`, val `0.5206`.
- Iteration 18: train `0.4219`, val `0.5139`.

The repeated AutoML pattern of rewriting `model.py` after each evaluation has not produced a stable downward validation trend. The important reference is the best iteration 7 result; later complexity generally drifted away from that result.

## Architecture Lessons From Latest Round

- Pre-LayerNorm (`norm_first=True`) appears important for stability; it coincided with recovery from validation loss around `0.88` to around `0.49`.
- Attention pooling appears to be the best useful architecture change so far; the best validation loss `0.4834` happened at iteration 7.
- Aggressive capacity scaling (`d_model` / feed-forward width / depth) caused instability or no reliable validation gain.
- Complex residual/skip variants, including global or sext-point skip patterns, did not improve validation and tended to degrade performance.
- Adding local temporal 1D convolution extractors after the best iteration did not produce a better result.
- Added complexity by itself is not currently translating into better validation loss. New architecture exploration is allowed, but it must be genuinely different from the failed/degraded paths and should be compared against the iteration 7 baseline.

## Avoid Repeating These Paths

- Do not keep rewriting architecture after every short run without a fixed baseline.
- Do not retry an approach that overlaps with a known failed or degraded approach unless the experiment explicitly explains what variable is different.
- Do not block all new architectures; block duplicated or weakly justified new architectures.
- Do not interpret lower training loss alone as progress; it has not reliably reduced validation loss.
- Do not assume more epochs will fix the issue unless the run uses early stopping and best-checkpoint saving.
- Do not compare architecture changes from single noisy runs only; use repeated seeds or controlled ablations where feasible.
- Do not continue adding layers, wider feed-forward blocks, complex skip paths, or local conv extractors without a controlled comparison against iteration 7.
- Do not change normalization, target masking, or output scale casually; those are core experiment-contract details.
- Do not introduce CES target leakage through `ces_history`.
- Do not judge performance only by aggregate normalized MSE when per-target `TI` and `VT` errors are needed.

## Recommended Next Experiments

Use controlled exploration instead of repetitive architecture churn:

1. Restore or preserve the newer round's iteration 7 baseline as the canonical best model.
2. Reproduce iteration 7 with fixed seeds and best-checkpoint saving.
3. Run the preserved baseline for 30-50 epochs with early stopping only after reproduction is confirmed.
4. Before making a change, check whether the idea duplicates a previously tried path. If it overlaps, skip it unless it is a controlled reproduction of the best iteration 7 baseline or has a clearly different mechanism.
5. Compare one change at a time:
   - learning rate and scheduler settings,
   - dropout and weight decay,
   - temporal subset augmentation on vs. off,
   - CES-history input on vs. off,
   - window size variants,
   - model variant that explicitly uses `input_mask`,
   - genuinely new sequence/spatial model alternative if it does not repeat prior scaling, skip-path, or local-conv attempts.
6. Track denormalized per-target validation error for `CES_TI` and `CES_VT`, not only aggregate normalized MSE.

## Useful Reference

`HANDOFF.md` contains the latest detailed session handoff and should be updated after major experiment rounds.
