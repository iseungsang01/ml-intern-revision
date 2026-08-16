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
  `analyze_gap.py`; 3-way split + `CES_INIT_SEED` in `train.py`; write-up
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

Deferred follow-up (was a standing rule in the retired AutoML loop's `program.md`): if input-defined
peak skill is weak or its CI straddles 0 (watch CES_VT), a **peak-weighted loss** (upweight
high-local-activity samples) is worth one controlled experiment. Never let peak metrics become the
selection gate — that stays the global mean `skill_vs_pchip`.

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

## Window Size — Settled Empirically (2026-08-04, two 24-run sweeps)

Full result in THESIS_RESULTS.md §8f (+ §8f-R contrast); runner
`ces_prediction/experiments/window_sweep/run_window_sweep.py`; data
`data/.wsweep_hf_summary.json` (held-free = the result) and `data/.wsweep_summary.json`
(held-kept = reference only). W ∈ {2,3,4,6,8} × 4 seeds + history-0, iter009 model,
per-run held-out TEST `skill_vs_pchip`.

- **One past observation is the whole story.** history-0 (`CES_ABLATE=no_history` at W = 4) puts
  `CES_TI` *below* PCHIP (−0.026, 0/4) and `CES_VT` at −0.783 (~1.8× PCHIP MSE). A single previous
  observation lifts both to their maximum at once (`CES_TI` +0.238 4/4, `CES_VT` +0.206). Fast
  diagnostics alone do not even reach interpolation parity.
- **Both targets are flat from history = 1 on.** `CES_TI` 0.190–0.246, `CES_VT` 0.190–0.206 across
  W = 2…8, while the per-point seed spread is 0.07–0.16 — wider than the whole curve. **W = 4 has
  no empirical justification and is below the W = 2/3 values on both targets.** The plateau rule
  returns W = 2.
- **The only real argument for W > 2 is coverage, not skill.** `compare_baselines.py` scores a
  sample only if window-persistence exists (an observed value *inside* the window), so W = 2 drops
  targets whose adjacent row lacks that reading. Totals barely move (`CES_VT` 52.1k → 53.2k) but
  the hard subset grows 4–10×: dt > 15 ms goes 456 → 1,958 and dt > 45 ms goes 14 → 135 from
  W = 2 to W = 8. Wider windows predict *more* long-gap samples, not better ones.

**The held-kept trap (why the first pass was wrong).** The runner originally popped
`CES_DROP_STUCK_TARGETS` to "inherit nothing", which silently selected train.py's default 0 —
violating §8c. That pass reported `CES_VT` rising +0.118 → +0.202 with W and concluded "V_rot needs
a long history". It was an artifact: the held-free gain decays monotonically with window
(+0.088 / +0.048 / +0.035 / +0.003 / +0.006 at W = 2/3/4/6/8), i.e. **held penalised short
windows** — a short window's one history slot is often a forward-filled copy, while a long window
still reaches a genuine reading. Remove held and W = 2 catches up completely; the slope vanishes.
`CES_TI` (~0.0 % held) shows no systematic shift. Lessons: **pin the data treatment explicitly in
every new run — never let it default**, and **any conclusion that scales with window length must be
re-checked held-free before it is believed**.

Two reusable mechanisms came out of this:

- **`CES_MAX_SAMPLES_PER_FILE`** (train.py, default 0 = off) caps each shot's samples with a seeded
  subset *before* the global caps. Mandatory for any window comparison: temporal-subset
  augmentation yields 240k samples at W = 2 but **30.1M at W = 8** (per-file max 187k), so an
  uncapped global subset is dominated by long-block shots more the larger W is. `sample_caps.json`
  in the split dir validates fixed-split reuse under capping.
- **`compare_baselines.py` honours `CES_ABLATE`** for the model's inputs only — persistence/PCHIP
  baselines keep reading the real history, so ablated points stay comparable on the same bar.

Verified before trusting the curve: the file-level split is **window-invariant** (probed W = 2/4/8
× 4 seeds), all 24 runs evaluate the **same 96 test shots per seed**, and the eval population
shrinks only 1.8% from W = 2 to W = 8.

**Windows trap for any long batch:** MKL's Intel Fortran runtime installs a console control handler
that aborts training when the parent console closes (`forrtl: error (200): program aborting due to
window-CLOSE event`, exit 3221225786 / 1073807364). Set `FOR_DISABLE_CONSOLE_CTRL_HANDLER=1` and
`KMP_HANDLE_SIGNALS=0` and launch detached
(`ces_prediction/experiments/window_sweep/run_detached.bat`); `--resume` absorbs the loss.

## Next Up (2026-08-04) — large-gap regime vs a CAUSAL baseline

Executed as §8g; the plan doc it referenced was removed once the batch landed. One-line version:
the dt-stratified sweep analysis shows `CES_TI` skill at dt > 45 ms is negative-to-zero at every
`W` (all CIs include 0, n = 429–505). That reads as "loses in the large-gap regime", but the
baseline there is PCHIP, which gets a **future anchor** — the very thing real-time nowcasting does
not have. `se_persistence` is not saved in `comparison_errors_test.npz`, so the causal-baseline
comparison cannot be run yet. **Add that one key to `compare_baselines.py`, re-run compare only
(no retraining, ~20 min for the 20 held-free runs), and re-judge.** If the model beats persistence
at large gaps, the limitation is "that regime belongs to interpolation", not a model weakness.

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

## Checkpoint / Architecture Provenance (2026-07-14) — READ BEFORE SCORING ANYTHING

The thesis result is **reproducible from source**, but the saved artifacts are booby-trapped.
Three traps, all verified:

1. **Fixed 2026-08-09 — `model.py` now IS the published architecture.** It used to be the
   Transformer the AutoML loop left behind, against which all 45 saved
   `weights/multimodal_ces.pth` files failed `load_state_dict`; every scoring script had to
   inject `model_iter009.py` over it first, and the runners did that by **copying the file over
   `model.py` on disk and restoring it afterwards**. `model.py` is now a thin re-export of
   **`ces_prediction/model_iter009.py`** (GRU + observation-masked multi-head attention; the
   iter2 "before" baseline is `ces_prediction/model_iter002.py`, both byte-identical copies of
   the AutoML archive, both SHA-256 pinned in `tests/test_architecture.py`). So
   `evaluate.py` / `compare_baselines.py` / `peak_analysis.py` load checkpoints out of the box,
   and an architecture variant is selected with **`CES_MODEL_FILE`** in the subprocess env
   instead of by rewriting tracked source. The lesson that outlives the fix: **a repo whose
   default import path is not the published architecture will silently score the wrong model** —
   it took a checkpoint-load failure to notice.

2. **`data/.improve_final_out/weights/` is NOT the checkpoint that produced
   `comparison_metrics.json`.** It reproduces CES_TI (RMSE 368.3 vs recorded 368.9) but gives
   CES_VT skill **+0.056** instead of the recorded **+0.161**. Do not trust it.

3. **Retraining the pinned architecture reproduces BOTH targets**, so the recorded thesis numbers
   were legitimate — only the weights files drifted. Symptom to recognise: **CES_TI reproduces from
   the stale weights on all 4 seeds, CES_VT does not (3 of 4)**. CES_VT has a wide CI
   ([-0.41,+0.34]) so a slightly different checkpoint swings its point estimate; CES_TI is stable.

**The whole presentation + THESIS_RESULTS.md are now unified on one reproducible checkpoint family**
(regenerated 2026-07-14; every number cross-checked to agree across `make_figures.py`,
`build_pptx.py`, and `THESIS_RESULTS.md`):

| what | dir |
|---|---|
| final model, seed 42 | `data/.vt_repro_out` |
| final model, seeds 1/7/123 | `data/.vt_repro_ms_{1,7,123}` |
| ablations | `data/.vt_repro_ab_{no_fast,no_history}` |
| iter2 "before" baseline | `data/.final_out` (**intact** — reproduces +0.0878 / RMSE 412.42 exactly) |

Headline (test, `CES_TI` skill_vs_pchip): **+0.257 / +0.194 / +0.263 / +0.280, all four PASS**;
`CES_VT` +0.154 / +0.109 / +0.065 / +0.127, all four n.s. Ablation (val, skill_vs_persistence):
fast-only `CES_TI` **+0.372**, fast-only `CES_VT` **−0.642** — the asymmetry mechanism.
Not every checkpoint is corrupt: `.final_out` is fine. Always verify a checkpoint against its own
recorded metrics before trusting it.

Two harness gotchas that cost hours:
- **Never hand-roll the eval loop.** `compare_baselines` drops ~160 CES_TI samples where a
  baseline is undefined; those few CES-fit-failure rows (CES_TI up to 15 keV) swing RMSE by 13%.
- **`train.py` rewrites `CES_SPLIT_DIR/split_manifest.json`.** With `CES_TEST_FRACTION=0`
  (the default) it writes a 2-way manifest and **destroys `test_files`** in a 3-way split dir.
  Back the manifest up, or point `CES_SPLIT_DIR` at a copy, before any training run.

## Framing Rules (2026-08-05, 승상님) — read before writing any results section

**"정보가 부족하다"는 결론이 아니라 변명이다.** The draft had drifted to
"information-limited, not capacity-limited" as a terminal claim. Negative results earn their place
**only** as routing information: they say where *not* to spend, which is what licenses a specific
claim about where the next gain is.

- **Never report a negative result without naming the measurement that would overturn it.**
- Worked example (paper §Headroom): the accumulated negatives (§8f window sweep,
  MC derived features, §8d seq) point at three levers, each grounded in this repo's own numbers —
  (1) history **reach**, not depth (§8i: only 54.1% `CES_TI` / 4.8% `CES_VT` of genuinely missing
  rows are even in-domain at W=4; §8f: W buys coverage, not skill); (2) the **Mirnov information was
  destroyed by preprocessing, not absent from the plasma** (§8b.2 lag-1 autocorr: BES +0.568 / ECEI
  +0.572 / **MC −0.009**, 82% of blocks |r|<0.1 → 100 Hz decimation of a kHz dB/dt with no
  anti-aliasing filter; fix is per-window RMS / band power / mode number from the **raw** stream);
  (3) the **NBI torque channel is absent** (§8b.3: `T_e`~`CES_TI` r=+0.353 vs `T_e`~`CES_VT`
  r=+0.024 — power is not torque).

**Novelty is stated as extension, never as absence (승상님 2026-08-05).** Do not lead with
"no prior work exists" — that clashes with how papers are written and collapses on a single
counterexample. The canonical order: acknowledge the active lineage (NN-CES fitting since JET
'93, cross-diagnostic inference, temporal densification) → state the three extensions (electron
→ sparse ion target; simultaneous memory-less → causal history-conditioned; assumed
reconstructability → pre-registered per-target tests) → close with ONE conventional hedge
("to our knowledge this conjunction has not yet been addressed; the family's natural next
step"). The extension frame gets stronger the more prior work you cite.

**Separate the two claims; never blur them.** "Beats future-using interpolation" is a statement
about the **observed** population. "Beats every causal method" is the statement that survives
reweighting to the **genuinely missing** points (§8i: 4/4 at +0.29 vs persistence; 1/4 vs PCHIP).
An online virtual sensor competes with persistence, not with an interpolant that reads the future.
Conflating them is the main way this result could be oversold.

## Numbers Must Come From Artifacts, Never From Prose (2026-08-05)

A full audit found the paper describing a **different architecture** than the one that produced its
numbers, and quoting a **superseded checkpoint family** throughout (§8h). Root cause: numbers were
transcribed by hand into `main.tex` and hard-coded as literals into `make_figures_en.py`, so
regenerating the checkpoints updated `THESIS_RESULTS.md` and nothing else.

Now enforced by construction:
`ces_prediction/collect_paper_numbers.py` → `docs/paper/paper_numbers.json` → read by
`docs/paper/make_figures_en.py`. **Never hard-code a number in a figure script or a paper table
without regenerating the JSON first.**

**Schema v2 (2026-08-16, after B.5).** The collector was rebuilt on the confirmed protocol: it reads
only the B.1–B.5 batch verdicts (`.b5_summary.json`, `.b1_gate_summary.json`, `.b2c_v3_summary.json`,
`.b3c_b3k8_summary.json`, `.b3_probe_summary_b3k8.json`, `.b4_scale_summary.json`,
`.wsweep_hf_summary.json`, `.protocol_audit_stats.json`, `.b5_spike_structure.json`,
`.latency_benchmark.json`) plus the per-run TEST reports for the RMSE ladders, carries BOTH
populations (`cut` / `incl`) in every block, and cross-checks B.5 ladder vs headline, B.3 vs B.5, B.4
width-160 vs headline (1e-4, CUDA drift) and report-vs-npz row counts before writing. The paper
(`main.tex`, 2026-08-16) and `make_figures_en.py` were rewritten against it, and the four presentation decks,
the 1-pager and `docs/presentation/make_figures.py` were rebuilt on it the same day (deck README records what changed).

A MiKTeX toolchain is now installed (`pdflatex`/`xelatex`/`bibtex` under
`%LOCALAPPDATA%/Programs/MiKTeX/miktex/bin/x64`); build `main.tex` with pdflatex+bibtex, `main_ko.tex`
with xelatex+bibtex, and treat a non-zero rc or any `!` line in the log as a failed build.

## Deployment Facts (2026-08-05, re-measured 2026-08-16 for the backbone)

- **The adopted seq_v2 backbone is run online *statefully*** (carry the LSTM (h, c) across the 10 ms
  grid; one recurrent step per row at batch 1 — `experiments/latency/bench_latency.py`, `seq_v2_step`).
  Idle laptop: **1.05 ms median / 1.61 ms p99 on CPU** (16% of the budget; a second idle run gave
  0.51 / 0.99 ms — laptop power states move absolutes up to 2×, never the ordering), 1.21 / 2.31 ms on
  CUDA. Whole-segment re-run (what eval_seq does): 2.9 / 5.6 ms per 100 rows, 6.4 / 8.9 ms per 300
  rows on CPU (35–47k rows/s). Windowed control at batch 1: 3.8 ms median but 18.9 ms p99 (W=2, CPU).
  Never benchmark with anything else on the machine — a concurrent training job inflated tails 5–10×
  and the artifact was annotated and re-run.
- **(2026-08-05, windowed model, W=4 era) Run the model on CPU, not GPU, for online inference.** batch-1
  p99 = 6.4 ms (W=4) / 8.7 ms (W=2) on CPU vs 21 ms median / 43–72 ms p99 on CUDA — an 8× penalty,
  because 201k parameters give kernel-launch overhead nothing to amortize against. Still the right
  guidance for the backbone (CPU step ≈ GPU step at batch 1; nothing to amortize).
- **Uncertainty without retraining: split conformal.** A learned variance/quantile head would move
  the point predictions and confound every skill number; conformal calibrates on val, changes
  nothing, and is distribution-free. Model intervals beat both baselines' 8/8 by Winkler score at W=4,
  and **32/32 cells (2 populations × 2 targets × 2 variants × 4 splits) for the seq_v2 backbone at W=2**
  (§8ab); in the inclusive population the model's T_i intervals are wider than PCHIP's and still score
  better because they miss less.
  Its real limitation is that coverage is **marginal, not conditional** — per-shot coverage runs
  50–100%, because calibration and test are disjoint discharges and shot-level shift breaks
  exchangeability. Report per-shot spread, never just the pooled number.

## Repeat Offender: The Unpinned Data Treatment

§8f cost a wrong conclusion; the **anchor runner had the same bug** (2026-08-05) and would have
paired a held-free arm against a held-kept control. Checklist for every new batch:
1. Pin `CES_DROP_STUCK_TARGETS` explicitly in the runner's env dict — never inherit, never pop.
2. Pin the file split with `CES_FILE_SPLIT_FROM` + a test-isolation check (drop_stuck shrinks the
   valid-file list and the seeded shuffle then repartitions).
3. Pair against a control trained under the **same** treatment (`.sf_iter009_s*` is held-free;
   `.vt_repro_*` is held-kept), and verify the scored populations match row-for-row.
4. When re-scoring frozen runs, **verify every pre-existing npz key reproduces bit-identically**
   before trusting an added key (the largegap/UQ re-runs do this and it caught nothing — which is
   the point: it is what makes the added key trustworthy).

## GP Arm + Fit-Failure Sensitivity (2026-08-05) — two audit follow-ups, both decisive

Full records THESIS_RESULTS §8p / §8q; artifacts `data/.gp_analysis.json`,
`data/.fitfail_analysis.json`, `comparison_errors_test__test_genuine_gp.npz` per run dir.

- **GP is the strongest offline arm and the model TIES it.** The harness's dormant GP arm
  (never ran — sklearn absent) is now an exact numpy Matern-3/2+white GP: nearest-16+16 local
  fit, per-sample grid-ML hyperparameters, deterministic, 0.94 ms/fit (the sklearn draft was
  38 ms/fit = infeasible). GP beats PCHIP +0.21…+0.28 (`CES_TI` 4/4 PASS). Model vs GP: 1/4
  PASS, 0/4 against, mean ≈ −0.01 → **tie**. Subsets (peak, dt bins) do not break it.
  **Never write "beats every offline interpolation"** — the honest form: "beats the
  pre-registered interpolants (PR1); ties the strongest future-using ML-tuned smoother; the
  causal/deployment claim is unaffected (GP needs future anchors)". Named tie-breakers: more
  shots (seed 7 resolves it), and a *causal past-only GP* arm (not yet run).
- **Fit-failure artifacts DEFLATE the headline.** Dropping `CES_TI` > 3 keV rows (0.4–0.6%)
  keeps 4/4 PASS and roughly doubles skill (+0.18…+0.28 → +0.36…+0.59): the spikes are
  unpredictable for every arm and drag the MSE ratio toward 1. The headline keeps them
  (pre-registered population) and is therefore **conservative** w.r.t. this artifact — this
  flips Q15 from a defended weakness into a strength.
- **Additive-key pattern held for the third time** (after §8g, §8i): adding the `gp` method
  cannot shrink `valid` because its NaN condition equals `ar_local`'s; all 4 re-runs reproduced
  the §8g npz bit-for-bit (+`se_gp`, `y_true` keys). `y_true` in the npz now enables further
  target-value-conditioned sensitivity analyses without re-scoring.

## `CES_VT` Is Three Regimes, Not One Verdict (2026-08-09) — §8r

The peak × held crosstab (evaluation only) decomposed the global `CES_VT` n.s. and the
answer overturned the standing hypothesis (from the since-removed `docs/ces_vt_proposals.md`):

- **Peaks are held-RICHER than the bulk on 4/4 seeds** (68/62/71/73% vs 58/51/48/46%), not
  genuine-richer as predicted. A forward-filled staircase (flat, flat, jump) *is* large local
  slope, so the input-only activity detector partly keys on the instrument's hold pattern.
  Any future peak work should recompute the detector from genuine neighbours only.
- **The three regimes behave completely differently** (skill vs PCHIP): genuine·peak
  +0.55…+0.63 (positive 4/4, PR4 1/4) and **+0.75…+0.82 vs persistence, PASS 4/4**;
  genuine·bulk ≈ 0 (interpolation is near-optimal on smooth stretches); held rows −48…−411,
  which is **structural, not a model failure** — PCHIP passes exactly through a value that is
  by construction the previous one, so no causal method can win there.
- **Dropping held rows lifts every seed (+0.154→+0.201 etc., 4/4) but never reaches PR4.**
  So held dilution is real but is not the whole reason `CES_VT` is n.s.; power over ~63 shots is.
- `CES_TI` (≈0% held) is the clean control and reproduces the peak concentration **4/4 PASS**,
  so the "edge lives in high-variability neighbourhoods" claim does not depend on the artifact.

Never again write "`CES_VT` is n.s. globally but PASSes at peaks" without the decomposition —
the global number is an average over regimes with opposite signs.

## Derive The Design From The Structure First (2026-08-09) — §8t

This problem is **latent state estimation under multi-rate sensor fusion**, not sparse-target
regression: one plasma state, observed densely/fast (BES/ECEI/MC) and sparsely/noisily (CES).
Four design decisions follow from that framing alone, and this repository found each one as a
separate controlled experiment over eight months:

| the isomorphism forces | we found it as |
|---|---|
| state exists at unobserved times → full grid + loss-side masking | §8d |
| a hold is not an observation | §8c |
| each discharge is an independent realisation → per-shot standardization | §8s |
| a physically-zero observation function must be structurally blocked | iter009 V_rot routing |

`seq_v2` assembles all four and **is the best `CES_TI` on 4/4 splits** (+0.255/+0.208/+0.305/
+0.308, all PASS; paired vs the held-free control 4/4 positive, mean +0.045, 1/4 significant),
**removes §8d's 4/4-significant `V_rot` deficit**, and trains in **1.2–1.4 min/seed against
12–22** — the window pipeline's cost is entirely inherited from the early decision to drop rows
with no observed target.

The `V_rot` repair was then attributed by a third arm (`seq_v2_nops`, routing on / per-shot off):
**4/4 significantly worse (v1) → 1/4 (routing) → 0/4 (routing + per-shot)**. So the routing is
the repair and per-shot standardization closes the last seed. **Block a channel at the encoder,
not the head** — a shared recurrent state carries the blocked information regardless of how the
head is wired, and that is why v1's `V_rot` head could not be fixed by reweighting.

The two methodologies are an **ordering, not a competition**: the controlled experiments are
what make each ingredient individually credible, but the framing was available on day one and
would have pointed at roughly the same design for a tenth of the compute. For the next problem:
**derive the candidate design from the structure, then spend controlled experiments proving its
parts** — do not start from the nearest supervised-regression template and recover the structure
through post-hoc diagnostics.

## Per-Shot Input Standardization Is Adopted (2026-08-09) — §8s

`CES_PER_SHOT_NORM=1` (z-score BES/ECEI/MC within each discharge; **targets untouched**) is the
repair §8n named for the campaign-transfer failure, and it works:

- **Campaign split: +0.155 mean paired `CES_TI` gain, 4/4 seeds, every CI excluding 0**, turning
  §8n's 0/4 vs-PCHIP PASS into 2/4. The base arm was *below* interpolation on 2/4 seeds.
- **Headline split: 0/4 significant losses** (mean −0.036; worst seed 42 −0.127 with CI upper
  bound +0.008). So the honest phrase is "no measurable cost", not "free".
- `CES_VT` is unmoved either way — the V_rot head never sees the fast diagnostics, which makes it
  a clean negative control for the mechanism.
- It also confirms §8n's diagnosis **causally**: if the campaign loss were physics changing between
  campaigns, removing an input *level* shift could not have recovered it.

Deployment caveat: an online estimator cannot see a shot's future to compute its σ, so per-shot is
the **offline upper bound** of this family; the deployable version is an expanding-window/EWMA
estimator and that gap is unmeasured.

## One Report Filename Per Split (2026-08-15)

`eval_seq.py` and `compare_baselines.py` wrote `comparison_metrics.json` for **whichever split
was scored last**. A val re-score of two frozen B.1 backbone dirs (B.2's `ensure_baseline_val`)
silently replaced their TEST reports, and two descriptive tables (§8z ladder, a session summary)
quoted val numbers as TEST before it was caught. The npz files were never touched, so no paired
verdict moved. Both scripts now write `comparison_metrics_{split}.json` always and the legacy
unsuffixed file only for TEST (or when none exists); the two reports were regenerated
(`experiments/b1_gate/regen_test_report.py`, population bit-identical, `se_model` drift bounded).
**How to apply:** read descriptive skill from `comparison_metrics_test.json` or recompute from
the npz; treat an unsuffixed report in a dir that also holds `comparison_errors_val.npz` with
suspicion; and always score val *before* test when both are needed in one dir.

## Bit-Identical Re-scoring Has a Limit: `se_model` (2026-08-09)

The §8g/§8i/§8p additive-key pattern ("re-run must reproduce the reference npz bit-for-bit")
**failed for the first time**, and the cause is not a population change: on this machine a
float32 CUDA forward pass is not bit-reproducible across sessions. Identical weights,
identical population, per-sample relative drift median 3 × 10⁻⁴, RMSE 372.3162 → 372.3135.

The fix, now implemented in `experiments/heldpeak/rerun_compare_stuck0.py` and the pattern to
copy: **split the keys by what they prove.**
1. Population keys (`shot`, `dt_ms`, `is_peak`, and the pchip/linear/persistence SEs) must be
   **bit-identical** — they are what "the scored population did not move" actually means, and
   they were bit-identical on all four seeds.
2. `se_model` gets a **bounded-drift** check (RMSE < 0.01 physical units) instead.
3. The merged artifact **keeps the reference `se_model`**, so no published number can shift
   underneath a re-scoring pass that was only supposed to add metadata.

## Direction Reset (2026-08-12, 승상님) — One Confirmed Protocol, Full Re-experiment

Decided 2026-08-12 on the Notion working page; recorded in THESIS_RESULTS.md §8v (2026-08-14).

- **Confirmed protocol for every new run: `W = 2` · held-free (`genuine`) · pre-registered
  `CES_TI` fit-failure exclusion** — the cut applies in three places (training targets, history
  inputs → treated as missing, evaluation population), identically for every arm; the
  spike-inclusive population is demoted to a sensitivity row (**amended 2026-08-14 by 승상님: the
  inclusive population is co-primary — an unqualified claim must hold in both**; §8ab below).
  Threshold re-justified from the
  current data's p99 (2,089 eV; > 3 keV = 1,197 rows = 0.53% of observed `CES_TI`), with
  2.5 / 3 / 4 keV sensitivity reported.
- **Every `W = 4` number is provisional and will be replaced.** Do not quote the draft's §5–§7
  numbers, and do not use a `W = 4` artifact as the control arm of a new confirmatory claim
  (historical reproduction excepted).
- Execution order: B.7 audit → B.1 backbone gate (`seq_v2` × 16 seeds + budget equalization)
  ∥ causal (past-only) GP → B.2 exploration (TEST frozen; selection on val only) → B.3 minimal
  interpretable model → B.4 scaling ceiling → B.5 full re-score; B.6 kHz-Mirnov async.
- **Why:** §8f, §8c and §8q each independently indicted the provisional protocol; replacing
  numbers piecemeal would leave the paper quoting mixed populations.
- **How to apply:** no experiment starts before the B.7 audit inventory is committed and its
  (B)/(C) corrections are folded into the pre-registration
  (`ces_prediction/experiments/PREREGISTRATION_W2.md`); TEST is scored once per confirmatory run
  with a decision rule fixed beforehand.

## Interpretability Is Structural, Not Linear — B.3 (2026-08-15) — §8z

Under the confirmed protocol the backbone is `seq_v2` (§8x, B.1) and its `T_i` skill compresses
**without loss** into `persistence + 8 bounded latents × a linear readout`
(`experiments/seq/model_seq_b3.py`, 21,498 params = 6% of the backbone): paired `T_i` vs seq_v2
mean **+0.002** on 4 TEST splits, PR4 PASS 4/4, causal GP PASS 4/4. What this changes:

- **The §8k "opacity price" (69% of the margin) was the price of the *window* form, not of the
  task.** The named-terms anchor+Δ, retrained at W = 2, collapses onto persistence (recovers 1% of
  the gap: its slope term needs two observed history rows and never fires; window statistics over
  one row are noise). Do not quote §8k's 31% under the confirmed protocol.
- **Recipe that worked**: exact decomposition (`ŷ = anchor + Σ w_k z_k + b`, zero-init readout so
  training starts at persistence), a small nonlinear causal encoder behind a **tanh bottleneck**
  (every number the readout sees is probeable), routing at the encoder. Linear probes on TEST
  inputs say the `T_i` latent encodes last-observed `T_i` (R² 0.5–0.75) and the ECEI `T_e` proxy
  (0.3–0.5), distributed across dimensions — local BES activity and staleness are barely linearly
  decodable.
- **Bounded corrections cannot recover a spiked anchor.** `V_rot` vs the backbone looks bad
  (−0.14…−0.55) but 0–4 rows per split whose previous `V_rot` observation is a fit-failure spike
  carry 28/0/64/72% of b3's `V_rot` squared error; excluding them the gap is −0.12/−0.14/0/0.
  `T_i` is immune only because the 3 keV cut removes such anchors. **`V_rot` has fit-failure
  spikes too** (§8q recurring) — a `V_rot` spike audit + cut/sensitivity rule is 승상님's call
  (B.7 follow-up); until then any persistence-anchored `V_rot` MSE comparison must report the
  spike-row share.
- **Training-budget trap for small models**: b3 was cap-bound at the seq family's 30 epochs while
  the backbone stopped by patience at 14–19; the like-for-like regime is "same stopping rule, cap
  non-binding for both" (100 here; runs terminated 54–88). Check `best_epoch == epochs_run`
  before reading any small-model comparison.
- **How to apply:** the interpretable rung of the thesis is b3k8, and it doubles as the ablation
  that says what the backbone's `T_i` skill *is* (persistence + a small correction from `T_e`-like
  state). The main model stays seq_v2 (§8x). B.4 (scaling ceiling) can now be read against a
  21k floor that already sits at the backbone's `T_i` skill.

## The Model-Size Axis Is Closed — B.4 (2026-08-15) — §8aa

seq_v2 `T_i`-encoder width 24 → 260 (34k → 879k params, one variable, everything else fixed,
same stopping rule, TEST once per point): mean `T_i` skill +0.230 / +0.236 / +0.235 / +0.236 /
+0.230, paired vs the 160-unit backbone within ±0.008 on average, w260 significantly better on
1/4 splits, no width down to 24 significantly worse on ≥ 3/4. `V_rot` (branch fixed) does not
move. **Why it matters:** with §8z (a 21k latent-bottleneck model equals the backbone) this
closes the architecture-size lever — the `T_i` information in {100 Hz BES/ECEI/MC + CES history
+ time} is exhausted by ~50k parameters of causal recurrent state; split variance (s42 vs s123)
dwarfs capacity effects. **How to apply:** do not propose a bigger/deeper model as a next step;
the remaining levers are inputs (NBI, kHz Mirnov, CES fit-quality metadata) and data treatment
(the `V_rot` spike issue in §8z). Read any future "model X is better" claim against the ±0.03
single-split noise seen here.

## The Two-Population Rule Earned Its Keep — B.5 (2026-08-16) — §8ab

Every W = 4-based analysis is now replaced by W = 2 · held-free numbers in **both** populations
(cut 3 keV / spike-inclusive), runner `experiments/b5_rescore/run_b5.py`, verdict
`data/.b5_summary.json`. What is now safe to quote, and what changed:

- **Unconditional (holds in both populations)**: backbone `seq_v2` `T_i` beats PCHIP 4/4 + 4/4
  (means +0.236 / +0.268), beats the causal GP 4/4 + 4/4, peak-stratum `T_i` 4/4 + 4/4,
  conformal Winkler best in 32/32 cells, `T_i` Δt > 15 ms pooled PASS, `V_rot` routing
  bit-identical under `no_fast`, history-0 collapse, cut threshold 2.5–4 keV immaterial.
- **The campaign-shift verdict flipped for the adopted model.** §8n (W = 4 window model) had
  the offline-superiority claim collapse under a temporal split. At W = 2 the *window* model
  still collapses (2/4 cut, 0/4 inclusive; per-shot standardization repairs it, §8s), but the
  **`seq_v2` backbone beats PCHIP 4/4 and the causal GP 4/4 in both populations** on a test block
  of shots that post-date every training shot. Quote it as one temporal block × 4 init seeds.
- **§8z's "21k = backbone" is cut-conditional.** In the inclusive population b3k8 loses
  −0.16…−0.21\* to the backbone and 3/4 to the window family; the ≈ 1% spike-anchor rows carry
  70–83% of *every* arm's `T_i` squared error there. Bounded corrections cannot recover a spiked
  anchor (the §8z `V_rot` mechanism, now on `T_i` in p100).
- **Why p100 alone would mislead — measured.** The window family's eval-time `no_fast` ablation
  loses −0.25…−0.43 in the cut population (its margin *is* fast-diagnostic information) but only
  −0.03…−0.09 inclusive, where a history-only model still beats PCHIP by +0.15…+0.23 because the
  interpolator's anchors are spiked. The p100 margin contains a spike-robustness component; the
  cut population isolates the fast-diagnostic contribution. Keep both; never headline p100 alone.
- **Coverage numbers to attach to any W = 2 claim**: PR2 fallback `T_i` 0.3–0.4% but `V_rot`
  40–44%; MNAR in-domain `T_i` 54–68%, `V_rot` 4–6% (the reweighted `V_rot` row is
  uninformative, not negative). MNAR-reweighted `T_i` vs PCHIP is 2/4 cut / 4/4 inclusive, vs
  persistence 4/4 everywhere.
- **`V_rot` stays unresolved vs offline interpolation** (1/4, 2/4), ahead of persistence 3/4 in
  both, 4/4 on the campaign block, and PASS in the Δt > 15 ms stratum in both populations.
- **Fit-failure spikes are one-sample events, and value cuts are one-sided**
  (`b5_rescore/spike_structure_audit.py`): > 3 keV = 951 runs, 85% single-row, median 13× its
  neighbours, but the cut catches only 19% of ≥ 2× upward outliers and none of the 4,965 dips.
  `V_rot` has 119 rows > 1,000 km/s in 16 shots (101 in one block of s31181). **Decided 2026-08-16
  (승상님): `V_rot` stays uncut** — no rule, no retraining; report the spike-row SSE share with
  every anchored `V_rot` comparison; the audit script is where to price a rule if fit metadata
  ever arrives. Same decision: two-population report stays co-primary; B.6 waits.
- **Operational**: the resumable stage runner + `run_step`'s fresh-artifact rule let a 6-stage
  batch be interrupted and resumed with zero re-runs; check `best_epoch == epochs_run` on every
  new seq run (7/8 cut-sensitivity runs and 2/4 campaign runs sat at the 30 cap).
- **How to apply:** quote B.5 numbers with the population named; an unqualified sentence in the
  thesis must be backed by both columns of §8ab's verdict table. Do not quote §8n's collapse for
  the backbone, §8z's ladder for p100, or §8i's W = 4 MNAR numbers.

## Useful Reference

`THESIS_RESULTS.md` §8 is the per-experiment record — add a section there after every controlled
round, and summarize lasting lessons into this file.
`docs/presentation/make_figure_transient.py` wires up the pinned architecture + trustworthy
checkpoint correctly and is the working reference for scoring the thesis model.

**Removed 2026-08-05** (recover from git history if ever needed): `automl_agent_loop.py` and its
`slack_notifier.py` / `program.md` / `HANDOFF.md` — the autoresearch loop that produced the thesis
architecture, unused since 2026-06-24; and the superseded docs `README2.md`, `AGENTS.md`,
`PROGRESS.md`, `RESEARCH_PLAN.md`, `RESEARCH_SUMMARY.md`, `ML_WORKFLOW_ARCHITECTURE.md`. The loop's
archived output was **kept** — and on 2026-08-09 it was copied into the repo as
`ces_prediction/model_iter009.py` (with the iter2 baseline as `ces_prediction/model_iter002.py`),
because `data/` is gitignored and the published architecture was therefore living outside version
control. `tests/test_architecture.py` pins both by SHA-256.

**Removed 2026-08-09** (recover from git history if ever needed), on the rule *the tree carries
what the current paper needs; `THESIS_RESULTS.md` §8 carries what we learned*:

- `ces_prediction/experiments/ct/` — the continuous-time encoder batch. Verified negative, and the
  paper's claims about it were removed at the same time, so the code backed nothing. **§8e still
  holds the verdict**, which is the part that stops anyone re-running it at `W = 4`. Its two
  reusable pieces were promoted rather than deleted: `experiments/runner_common.py` (the split /
  control / env constants that ten batches were importing *from the CT runner*) and
  `experiments/paired_model_compare.py`.
- Executed plan documents whose outcomes are now recorded as results: `docs/연속시간_모델_실험계획.md`,
  `docs/설명성_피드백_실험계획.md`, `docs/ces_vt_proposals.md`, `docs/정비_실행계획_2026-08-09.md`.
- Literature-search intermediates under `docs/paper/litreview/` (candidate/enriched JSON, the
  harvest scripts, the raw bib pool). The judgement, `NOVELTY.md`, was kept and moved to
  `docs/paper/NOVELTY.md`; `refs.bib` is the surviving bibliography.
- The tracked `.omc/` plans, specs, drafts, and research notes — agent scratch describing work that
  §8 now records as outcomes. `.omc/` is gitignored in full.
- LaTeX build artifacts (`.aux/.bbl/.blg/.log/.out`), now gitignored. `main.pdf` / `main_ko.pdf`
  stay tracked: they are the deliverable, not a build product.
