# Project Knowledge

This file records prior attempts, known constraints, and directions to avoid so future work does not repeat the same failed paths.

## Confirmed Thesis Result (interpolation comparison)

**The model beats conventional past+future CES interpolation for CES_TI, robustly; not for CES_VT.**
Final model = AutoML-improved "iter5": GRU history-encoder + multi-head attention per-target heads
(window=4), in `ces_prediction/model.py`. Evaluated on a held-out **test** split (3-way
train/val/test; selection on val only) with per-target physical-unit RMSE + `skill_vs_pchip` and a
**shot-clustered paired bootstrap** (the shot is the resampling unit — rows within a shot are
autocorrelated).

- Across 4 independent held-out splits (seeds 42/1/7/123): **CES_TI skill_vs_pchip = +0.20…+0.30,
  95% CI excludes 0 every time (PASS)** vs both PCHIP and linear; **CES_VT n.s. on all four.**
- Physics reading: fast diagnostics (BES/ECEI/MC) carry ion-temperature info beyond temporal CES
  interpolation (collisional e–i coupling); they carry ~no toroidal-rotation info (NBI torque
  unobserved; Mirnov aliased at 100 Hz) → V_rot stays history-autocorrelation-limited. This is the
  T_i↔V_rot asymmetry, now confirmed against a strong (future-using) interpolation bar.
- The earlier baseline (iter2) was n.s. (+0.088) on its split; optimizing the loop on val
  `skill_vs_pchip` (not skill_vs_persistence) produced the significant iter5.
- Limitations: skill on observed CES points only (MNAR optimistic bound); window=4; offline
  comparison (baselines use future CES, the model does not — beating them is the strong claim).
- Tooling: `compare_baselines.py`, `bootstrap_compare.py`, `baselines_interpolation.py`,
  `analyze_gap.py`; 3-way split + `CES_INIT_SEED` in `train.py`; running log `PROGRESS.md`; write-up
  `THESIS_RESULTS.md`; baseline citations `docs/interpolation_baselines_references.md`.

## High-Variability (Peak) Reconstruction Finding (2026-06-23, GPU)

A layered analysis (no retraining, no model/contract change) of *where* the uniformly-trained
model earns its edge over interpolation, via `ces_prediction/peak_analysis.py`. "Peak" = high
**local-activity NEIGHBORHOOD** flagged from a **target-independent input proxy** (Family i-a
`ces_activity`: large neighbor-bracket slope and/or high local CES-neighbor variance, computed
from `build_neighbor_set` which **excludes the target row** — so it never reads `CES[idx]`, a
non-circular headline). NOT pointwise extrema. Headline peak SEs are byte-identical to the global
`compare_baselines.py` SEs (one additive npz key `{name}_is_peak`, sliced by `valid`). Shot-
clustered paired bootstrap (B=10000, seed 12345); dual-sufficiency guard (`N_MIN_PEAK_SHOTS=15`
binding, `N_MIN_PEAK_ROWS=200`). Val split, observed-CES-only (MNAR optimistic bound), physical units.

**Headline result (default params, val, 8000-sample eval, run dir `data/.improve_final_out`):**

| Target | Global skill_vs_pchip | Peak (i-a) skill | Peak 95% CI | Peak verdict | rows / shots |
|--------|----------------------:|-----------------:|-------------|:------------:|-------------:|
| CES_TI |                +0.515 |          **+0.855** | [+0.658, +0.933] | **PASS** | 965 / 119 |
| CES_VT |                +0.241 |          **+0.691** | [+0.107, +0.871] | **PASS** | 1760 / 82 |

1. **The model's edge over interpolation is concentrated in high-variability neighborhoods.**
   For CES_TI, peak skill (+0.86) far exceeds global (+0.51): interpolation is near-optimal on the
   smooth bulk, and the model's real value is in the active stretches.
2. **MAJOR / surprising: CES_VT PASSES at peaks (+0.69, CI lower bound +0.11 > 0) even though the
   GLOBAL CES_VT result is weak/borderline (n.s. on held-out test in the thesis).** So the
   T_i↔V_rot asymmetry is *regional*: averaged globally the model barely beats interpolation on
   V_rot, but in high-local-activity neighborhoods it does — where smooth past+future interpolation
   is worst and the (history-driven) model has the most to add. Report this honestly as a regional
   strength, not a reversal of the global n.s. test (it is on the optimism-caveated val split).
3. **AC7 ablation (flag-gated `CES_PEAK_ABLATION=1`, `no_fast` zeros BES/ECEI/MC), pinned sign
   convention `paired_diff = SE_no_fast − SE_full`, significance iff `paired_diff_ci95[0] > 0`:**
   - **CES_TI: significant** (diff CI ≈ [+2066, +36997] ≫ 0) — removing fast diagnostics hurts a lot
     at peaks ⇒ the model genuinely uses the multimodal (BES/ECEI/MC) signal for T_i in active regions.
   - **CES_VT: NOT significant** (diff CI = [0, 0]) — the `no_fast` arm is identical at V_rot peaks ⇒
     V_rot peak skill is **history-driven, not fast-diagnostic-driven**. Consistent with the
     2026-06-22 input-modality ablation. (OOD caveat: zeroing inputs is off-manifold; disclosed.)
4. **Sensitivity sweep (3 settings) — robust.** slope_z/neigh_var_pct = (2.5,0.10 default),
   (3.0,0.05 stricter), (2.0,0.20 looser): CES_TI peak skill +0.83…+0.85, **PASS all three**;
   CES_VT +0.54…+0.71, **PASS all three** but with a thin CI lower bound (+0.01…+0.11), so the
   CES_VT-at-peaks result is real but fragile. Default (2.5, 0.10) gives the strongest CES_VT CI
   lower bound and is the pinned choice.

Implications for the AutoML loop: `program.md` now carries a standing peak-steering rule — if
input-defined peak skill is weak or its CI straddles 0 (watch CES_VT), the researcher may propose
a **peak-weighted loss** (upweight high-local-activity samples) as one controlled experiment. This
is the deferred non-goal; the keep/discard gate stays the global mean `skill_vs_pchip` (peak metrics
inform the researcher only, never the gate). See [[ces-thesis-result-confirmed]].

## Current Status

- Primary reference point: best validation loss `0.4834` at iteration 7 of the newer architecture-search round.
- Best-known train loss at that point: `0.4494`.
- Use the iteration 7 design as the baseline to preserve, reproduce, and tune.
- Later runs are mainly evidence about what moved away from the best result, not better starting points.
- The project should preserve the best baseline while still allowing genuinely new architecture exploration that does not duplicate failed paths.
- Note: the architecture-search numbers above are a separate track from the **input-modality
  ablation finding (2026-06-22)** below, which is about *which inputs carry signal*, not architecture.

## Input-Modality Ablation Finding (2026-06-22, GPU)

Controlled ablation on the **model input modalities**, on real data (40k train / 8k val,
12 epochs, batch 1024, window 4, temporal subsets on; shared file-level split; RTX 5060,
torch 2.11.0+cu128). The three variants share identical settings — only the model inputs
differ via `CES_ABLATE` (which zeroes a modality group). The **persistence baseline is
always computed from the real history**, independent of the ablation.

| Variant (model inputs)             | CES_TI skill | CES_TI R² | CES_VT skill | CES_VT R² |
|------------------------------------|-------------:|----------:|-------------:|----------:|
| Full (history + fast + time)       |       +0.359 |     0.347 |       +0.180 |     0.788 |
| no_fast (history only)             |       +0.393 |     0.382 |       +0.234 |     0.802 |
| no_history (fast diagnostics only) |       +0.162 |     0.146 |       −3.31  |    −0.116 |

(skill = 1 − MSE_model/MSE_persistence; denormalized physical units; observed val points only.)

1. The full model **beats persistence on both targets** (TI +0.36, VT +0.18) — genuinely
   useful. (An earlier 1-epoch smoke that showed negative skill was just undertraining.)
2. **V_rot skill is entirely from CES history.** Fast-diagnostics-only scores −3.31 / R² −0.12
   (worse than predicting the mean); history-only matches/beats the full model. BES/ECEI/MC
   carry ~no toroidal-rotation information at the 10 ms grid.
3. **T_i is different: fast diagnostics DO carry T_i info** — fast-only still beats persistence
   (+0.162). This is the physics-predicted T_i↔V_rot asymmetry, confirmed empirically.
4. **Fast diagnostics are redundant given history for both targets** (Full ≈ no_fast, and in
   fact consistently slightly lower). With history present, the fast inputs add ~0 (or marginally hurt).

Physics backing (web-grounded review, same day): T_i is constrained by collisional electron–ion
coupling (t_ei ∝ T_e^{3/2}/n_e), so ECEI(T_e)+BES(n_e) carry T_i info; V_rot is set mainly by the
**unobserved** NBI torque, and the Mirnov coils are **raw signals sampled to 10 ms (100 Hz)**, which
aliases away the kHz mode-rotation frequency that would otherwise proxy rotation. So the fast inputs
physically *should* carry T_i but not V_rot — exactly what the ablation shows.

Caveats: single seed, 40k samples, 12 epochs. The small Full-vs-no_fast gaps (~0.03) may be seed
noise (the −3.31 V_rot fast-only effect is far beyond noise). Evaluated on **observed** points only;
CES missingness is MNAR (drop-out at low S/N, ELMs, transitions), so observed-point skill is an
**optimistic bound** for the genuinely-missing points the model is meant to fill.

## Data And Model Contract

Every generated or edited `model.py` must preserve this interface and data contract:

- `model.forward` must accept `forward(self, bes, ecei, mc, time_features=None, ces_history=None)`.
- Outputs must be normalized `[CES_TI, CES_VT]` with shape `(batch, 2)`.
- Do not denormalize inside `model.py`; inverse transforms belong in evaluation/reporting code.
- BES, ECEI, MC, and targets use train-file-only per-channel z-score normalization (target stats NaN-aware).
- CES_TI and CES_VT are missing independently; rows are kept when inputs are complete and at least one CES target is observed, with a per-target `target_mask` and per-target masked MSE.
- `ces_history` has shape `(batch, window, 4)` containing normalized previous `CES_TI`, normalized previous `CES_VT`, `CES_TI` observed flag, and `CES_VT` observed flag.
- The target timestep is fully masked (both values and both observed flags `0`) in `ces_history` to avoid leakage.
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
