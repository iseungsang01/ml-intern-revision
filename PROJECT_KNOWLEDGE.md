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
  rows on CPU (35–47k rows/s). Windowed control at batch 1: 3.8 ms median, p99 **18.9 ms in that
  session but 4.46 ms when re-measured on 2026-08-17** — treat the 18.9 as contaminated and see the
  ordering rule two bullets down. Never benchmark with anything else on the machine — a concurrent
  training job inflated tails 5–10× and the artifact was annotated and re-run.
- **Every causal arm fits the 10 ms budget; the backbone wins on skill-per-millisecond, not on
  feasibility** (2026-08-17, §8ac, `experiments/reach/bench_causal_arms.py` + a same-session
  re-run of `bench_latency.py`; GP measured on the *real* TEST neighbour sets because its cost
  scales with neighbour count). `gp_causal` refits a Matern-3/2 GP per row over a 5 × 4
  hyperparameter grid: **1.14 / 2.34 ms (23%)** for skill vs persistence +0.319 `T_i` / +0.276
  `V_rot`. The backbone's stateful step: **0.97 / 1.49 ms (15%)** for **+0.396 / +0.390** — higher
  skill, 1.6× lower tail, and it *ties the acausal GP* using past data only. Window control:
  4.46 ms p99 at `W = 2` (45%), 5.05 at `W = 4` — **3.0× the backbone, but inside the budget**.
  Cheap arms: persistence 0.034 ms, `ar_local` 0.045 ms but skill **−4.04** (far worse than holding).
- **Batch-1 latency tracks OPERATOR COUNT, not parameters or FLOPs** (2026-08-17, §8ac §3).
  The window model does 274k MACs per step — **0.11 ms of arithmetic at 5 GFLOP/s, 3% of its
  measured 3.56 ms**; the rest is per-op dispatch across its **57 leaf ops**. `seq_v2`'s stateful
  step has **10**, does 1.29× *more* multiplies, and runs 3.5× faster. Consequences: (a) `W = 2`
  and `W = 4` have the **same 201,258 parameters** — the window sets sequence length, not weights;
  (b) shrinking widths barely helps (window −40% params → −21% ms; `seq_v2` −90% params → −17% ms),
  and the smallest window variant is still 2.7× slower than the largest `seq_v2`; (c) the lever
  that works is graph fusion — `torch.jit.trace` + `freeze` gave **1.73×** on the window model with
  identical outputs. Never argue about deployment cost from parameter counts on this project.
- **Latency absolutes on this laptop are not reproducible to better than ~4×; only the ORDERING is.**
  §8ac's first draft quoted the frozen §8l `W = 2` p99 of 18.9 ms and concluded the window family
  misses the deadline; a same-session re-measure gave 4.46 ms and refuted it. The frozen value was
  already physically suspect (`W = 2` at 2.3× the p99 of `W = 4`, though it walks the shorter
  sequence). **Never state a deadline verdict from a benchmark measured in a different session**
  — re-run every arm together, and phrase conclusions as orderings.
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

## Size Is Not The Mechanism — Structure Is (2026-08-17) — §8ad

`experiments/wslim/`: the window model re-derived for the `W = 2` input (flatten instead of a
receptive-field-5 conv stack over 2 steps; no `AdaptiveAvgPool1d(1)` averaging the 2 steps away;
no bidirectional GRU + 4-head attention over 2 steps). **25,602 params / 21 leaf ops / 0.66 ms vs
201,258 / 57 / 3.02 ms** — the shrink is real, and it is the *structure* axis, not the width axis
§8ac measured. But paired against the frozen B.1 control on the same rows: **`T_i` −0.087, 4/4
significant deficits** (below PCHIP entirely on s42), while **`V_rot` is indifferent** (+0.005,
0/4 either way).

- **The `V_rot` half of `iter009` is dead weight**, exactly as §8ab's routing predicts — a carried
  value plus staleness needs no sequence machinery. The `T_i` half is not.
- **Reconcile with §8aa/§8z**: skill is flat across a 26× *width* range and a 21k latent matches
  the backbone, yet 25.6k here loses 0.087. Capacity was never the binding constraint; the
  functional form is. Do not read "size doesn't matter" as "shape doesn't matter".
- **How to apply:** when a cost argument says "make it smaller", ask *which* axis. Width is free
  (and buys almost no latency); structure buys latency and can cost skill. And check whether
  latency binds at all first — §8ac says it does not here, so W-SLIM is measured, recorded, and
  **not adopted**.
- **Named follow-up** (the deficit is unattributed — three things changed at once): restore only
  the sensor conv stack, then only the history GRU+attention, and run W-SLIM at
  `sensor_feature_dim = 96` to separate structure from width. The sensor conv path is the better
  first guess, given `V_rot`'s indifference to the history machinery.

## The Reach Axis, Measured By Truncation (2026-08-17) — §8ac, **narrowed by §8ae**

> **Read §8ae first (2026-08-19).** This section's headline used to read "…And It Refutes The
> Window Framing". It does not. Every number below comes from **truncating one trained model**,
> and a model *trained* at reach 2 recovers **+0.260 of the 0.310** that truncation cost on `T_i`
> (4/4 significant, identical rows, `experiments/reach/trained_vs_truncated.py`). **At least 84%
> of the `ctx = 2` deficit is cold-start warm-up, not missing information**, and what full-block
> context is worth is bounded above by §8x's +0.081 — not −0.310. `CES_VT` is the internal control
> and behaves as §8ab's routing predicts (truncation costs it 0.032, training recovers none of it,
> because its branch rides carried input channels rather than the recurrent state).
> **How to apply:** never quote a truncation rung as an information statement. Quote it as "this
> model loses X when its state is thrown away".
>
> **B.9 axis A has now measured it (§8af, 2026-08-19).** `seq_v2` trained AND scored at reach
> 2/7/15/31/63: **both targets saturate at 7 steps = 70 ms**, not 500 / 200 ms. The truncated
> ladder needs **63** steps to match itself where a trained one needs **7** — a factor of 9, all
> cold start. Rungs 31 and 63 sit *nominally above* the unbounded backbone, so carrying state
> across a 1,482-step block buys nothing. §8x's +0.081 backbone-over-window gap decomposes into
> **reach −0.065 + architecture −0.016**. Quote 70 ms, not 500 ms.


`experiments/reach/`, no retraining: the frozen B.1 checkpoints re-scored with the LSTM state
reset `ctx` steps before every row (`CES_SEQ_CONTEXTS`; `ctx = full` reproduces the frozen
`se_model` **bit-identically** 4/4, which is what licenses reusing the population).

- **`T_i` uses 50 contiguous steps (500 ms); `V_rot` uses 20 (200 ms).** Beyond that the worst
  paired deficit across four splits is < 0.002 skill.
- **`ctx = 2` — the contiguous context a `W = 2` window model sees — loses to `gp_causal` by
  −0.34 on `T_i`, and `ctx = 1` is *worse than persistence* (−0.28).** §8f's "`W = 2` is enough"
  was about how many past CES **observations** to staple on; it never measured the dense-diagnostic
  context, which is where the `T_i` skill lives. Do not cite §8f as evidence that a short window
  suffices — cite it only for the history-length question it actually asked.
- **Read significance with an effect-size floor when arms are paired row-for-row.** `CES_VT` on
  split 7 holds a *significant* deficit of −5e-7 out to ctx = 300. The runner reports both the
  strict rule and a 0.002-skill practical floor; quote the floor-based saturation.
- **How to apply:** "recurrent vs. windowed" is **not** settled by this section (see §8ae above);
  §8x settles it on the combined reach+architecture gap (+0.081 pooled), not on reach. The open
  arms — a 63-step-receptive-field causal TCN and a causal-attention arm at the same reach — are
  pre-registered as B.9 axis B. Any such arm goes through the B.2 rule: val-only exploration →
  pre-registered decision → TEST once, ≥3/4 significant.

## Published Shots Are V_rot-Poor Shots (2026-08-18) — a selection criterion that fights the target

Asked whether the six μs-fetch shots that carry no publication could be swapped for published
ones, the data gate answered instead of an argument. Of the **nine published shots we hold a CSV
for, exactly one passes** — #31921. Seven of the eight rejections are the *same* rejection,
`vt_clean_n >= 60`: #31873 and #31923 have **one** independent V_rot value each, #31097 one,
#31888 one, #32027 eight, #31276 ten, #31357 forty-four. (#31359 fails elsewhere, on
`ti_clean_frac`.)

The discharges that get written up are edge/MHD experiments — FIRE mode, ELM suppression, error
fields, disruption — and in those the CES rotation fit is largely absent or held. Whatever the
instrumental cause, the consequence is a rule for future selections: **selecting shots for
publication status selects against V_rot supervision**, the target this project is already
weakest on. Publication buys external validation (a figure whose physics an independent group
already characterised), not training value, and the two must be traded explicitly, never
conflated. The `hires_shots` list therefore admits #31873 / #32027 on published physics *while
recording that they fail the gate*, and keeps the six data-picked shots.

Related, from the same pass: per-shot skill vs PCHIP correlates **+0.27 with `ti_std`** and
**+0.23 with `ti_transient`** for CES_TI (the model earns its keep where T_i actually moves;
flat stretches are where interpolation is already optimal) and **−0.31 with `vt_std`** for
CES_VT (it loses *more* where rotation moves). n = 288 / 179 shots, shot-level aggregates, so
this is structure to test at the segment level, not a headline. It is also the sharpest
statement yet of why NBI torque is the missing input rather than a missing layer.

## Searching The Fusion Literature: What Actually Works (2026-08-18)

Four routes were measured on this repo's own known-positive shots, which is the only way to
tell a real absence from an unreachable source:

- **OpenAlex `fulltext.search:KSTAR <shot>`** — the strongest tool. 4/4 on positive controls,
  and it reaches papers no downloader here can (it returned the IOP article `adacfc`, which has
  never once been fetchable, independently corroborating a hand-read claim). Caveats: OpenAlex
  meters the API (~a few hundred requests/day free, `Resets at midnight UTC`), so a 641-shot pass
  must be resumable; and `title_and_abstract.search` sees only 456 works where `fulltext.search`
  sees 2692 — the narrow filter is what hid the Nature ELM paper from the corpus.
- **Nature/Springer** — `nature.com` HTML and `media.springernature.com` SI files download fine.
  Supplementary material is worth crawling: it named three discharges (#31184/#31185/#31189)
  that appear nowhere in the main text.
- **IOP, AIP, Elsevier** — IOP serves a Radware CAPTCHA (to WebFetch, to scripts, and to browser
  automation alike), AIP returns 403. **Do not try to work around the CAPTCHAs.** The papers are
  frequently open access: sampling 60 unread relevant works found 38 OA. The barrier is a bot
  wall, so "we could not download it" must never be written as "it is not in the literature".
- **NASA ADS full-text search** — looked promising and is *not* usable: the positive control
  returned 0 results, then the site switched to a human-verification page. Any zero it produced
  is meaningless. Always run the positive control before believing a negative.

## The Family Ties Only Where Capacity Is Ample (2026-08-19) — §8ag + §8ai together

Two results that read as opposites are one statement, and quoting either alone is wrong.

- **At ~300k parameters the operator is irrelevant** (§8ag): recurrence, dilated convolution and
  banded attention trained at the *same* reach land inside ±0.02 of each other with no arm reaching
  3/4 significance in either direction.
- **Below 10k it is not** (§8ai): each dilated-conv arm beats the size-matched recurrent arm on
  `T_i` by **+0.027 … +0.040** with **3–4/4** significance — pre-registered H6, falsified.

The reconciliation: **the family does not decide the ceiling, it decides how many parameters you
need to reach it.** Every small TCN still lands within ±0.007 of the *full-width* LSTM at the same
reach and ±0.012 of the 357k backbone. The concrete consequence is the parameter floor that keeps
4/4 against the causal GP: recurrent **3,898** (`v2m4k`), convolutional **1,808** (`tcn2k`) — 2.2×
smaller, and tied to the backbone (+0.001 paired, 0 significant losses) where `v2m2k` was −0.036
with 3.

**The tie has a second bound, and it is reach (§8ak, 2026-08-20).** §8ag's arms were all trained
at 150 ms or 630 ms, so it never observed the threshold rung it was generalising over. Trained at
70 ms: recurrence and dilated convolution turn at the **same** rung (3/4 at 30 ms, 4/4 at 70 ms)
and tie paired at both (−0.004, no significant split), while **banded attention is 0.023 below the
same-reach LSTM with 3/4 significance** — `differs`, the only such verdict in B.9 — and does not
reach 4/4 until 150 ms. So the 70 ms threshold is **not** an LSTM artifact, and the arm that needs
the most context is also the one that costs the most per step (§8aj).

**How to apply.** Never write "the architecture does not matter" without **both** qualifiers:
"above ~10k parameters **and** above 150 ms of context, it does not." When a minimal model is
wanted, start from the convolutional arm. Nothing on this problem recommends attention. And
`V_rot` supports none of this — no rung of any family reaches 3/4, and the one large number
(+0.303) is against `b3m1k`, which varies family, size and branch structure at once, so it is
reported and never used.

**And state which bar you are reading.** At the project's own promotion criterion (≥3/4) all three
families clear at 70 ms; at the stricter 4/4 the attention arm needs 150 ms. The two readings
differ by one rung, so a sentence that omits the bar is ambiguous rather than wrong.

## Cost Is Dispatched Operator Count — And This Machine's Milliseconds Are Not Evidence (2026-08-19) — §8aj

`experiments/b9_latency/op_count.py` counts the `aten::` operators one online step dispatches. It
is the statistic to quote, because it is exact, reproducible on any machine, and it does not move
when something else is running:

- **recurrence — O(1) in reach.** 111 ops fused / 161 lean, identical at reach 2 and reach 630.
- **dilated convolution — O(log R).** Exactly **+48 ops per layer** (113/161/209/305 at
  L = 1/2/3/5), `RF = 2^(L+1) − 1`.
- **banded attention — O(1) in reach, 4.3× the constant.** 473 fused at bands 7, 15 and 63 alike.
- **the conversion is nearly constant**: min-median ÷ op count = **2.1–3.2 µs** across a 6.6× op
  range, a **151× parameter range**, and all three families. Stock `nn.LSTM` is the one outlier
  (6.7 µs/op) because module call protocol is not an ATen op — that gap *is* what a lean rewrite
  buys (2.10×, plus 1.07× from four exact fusions).

**And the timing side failed, correctly.** The 5-session re-run recorded a **21.84×**
session-to-session p99 spread (against 2.56× on the pass §8ah recorded), so the pre-registered rule
refused to resolve 1 ms — and its artifacts overwrote the ones §8ah's table was built from, which
is why that table is **suspended rather than corrected**.

**How to apply.**
1. **Never run a latency session next to a training batch or a network job.** That is the only
   difference between the 2.56× pass and the 21.84× one.
2. When a pass is contaminated, quote the **minimum** over sessions — noise only adds, so the
   minimum is the closest estimate of the machine — and label it a *lower bound*: fine for ordering
   arms, never sufficient to grant a budget pass.
3. A benchmark that overwrites its own per-session artifacts destroys the record behind any
   conclusion already drawn from them. Version the session files, or expect to suspend a section.
4. The remaining lever on cost is the **runtime**, not the model: 1,808 parameters did not make the
   milliseconds follow, because ~111 dispatches did not become fewer.

## The Shot Papers Anchor Our Settings (2026-08-21) — read before writing physics justifications

The five papers behind the frozen twelve were read in full (`data/shot_papers/`; the
setting-by-setting mapping lives on the Notion page "실험 설정의 물리적 근거"). The anchors
that outlive this pass:

- **CES missingness is operational, not random.** The I-mode/FIRE paper (NF 65 036003)
  states ion temperatures are "only available between two sequential beam blips" — the
  MNAR treatment, held-removal, and the beam-phase fetch windows all rest on this
  published sentence, not on our inference.
- **The T_i headline has a non-ML precedent in these very discharges**: the same paper
  shows BES fluctuation spectra (10–30 kHz drop = WCM prominence) correlating with the
  edge T_i pedestal height over 8 discharges × 672 time points. Fast diagnostics carrying
  T_i information is measured physics there, learned physics here.
- **The V_rot routing chain is published end-to-end**: rotation is NBI-torque-driven
  (same paper), magnetic non-axisymmetry exerts NTV torque that shows up as CES rotation
  braking (error-field paper, GPEC calculations), and the mode band that carries it is
  kHz (EHO 4/8/12/16 kHz on Mirnov, NF 66 096006) — i.e. exactly what 100 Hz aliasing
  destroys. B.6's H1 is the closure of this chain, not a hunch.
- **Every B.6 band edge is a published structure in these discharges**: 0.1–3 (zonal
  density tail), 3–10 (EHO n=1,2), 10–30 (I-mode intermediate band + EHO n=3,4),
  30–70 (WCM ~50 kHz), 70–300 kHz (PanoMHD L/H cross-power). Feature methods too:
  RMS-envelope amplitude (EHO paper), two-channel cross-power/cross-phase (PanoMHD,
  NFFT 512; error-field paper's v⊥ from cross-phase slope). **Raw MC is sampled at
  2 MHz** (PanoMHD, citing Bak & Lee 2001) — the expected value for the on-arrival slot.
- **Shot-level held-out evaluation is the field's own convention**: PanoMHD freezes a
  101-shot test set; ML runs at ms scale inside the KSTAR control loop (Nat. Commun. 15
  3990). Our shot-clustered design and 10 ms budget framing match practice, not just taste.
- **Three settings the literature does NOT back** (say so in the paper): per-shot input
  standardization (our measurement only), the physical identity of the 50 ms saturation
  (the EHO paper's 50 ms CES averaging is a suggestive coincidence, not causal evidence),
  and the 0.5 coherence threshold (statistically justified, not literature-derived).
- **Caution for the 3 keV cut's defense**: published core T_i reaches 5–10 keV in these
  same discharges. The cut is a *distribution* statement about our edge-side channel
  (p99 = 2,089 eV), never "3 keV is physically impossible" — phrase it that way or an
  examiner armed with the FIRE papers will puncture it.

**Second pass (same day): external anchors, all Crossref-verified, all in `refs.bib`.**
The NN-CES lineage's origin is literally Christopher Bishop (Bishop/Roach/von Hellermann,
PPCF 35, 1993; then Svensson 1999). The nearest living relative of our nowcaster is
**RTCAKENN** (Shousha et al., NF 2023, 10.1088/1741-4326/ad142f): real-time NN
reconstruction of DIII-D kinetic profiles *including carbon ion temperature and rotation*
from real-time-available inputs. The mode-rotation-frequency feature's physics is standard
theory (Fitzpatrick, NF 33 1049 (1993); La Haye, PoP 13 055501 (2006): modes co-rotate
with the flow at their rational surface, the frequency lands on Mirnov), and KSTAR has
dedicated NTV braking observations (Park et al., PRL 111 095002 (2013); Kim/Choe, NF 57
126035 (2017)). The B.6 coherence-validity rule is exactly the textbook coherence
significance level P(γ² > c) = (1 − c)^(n−1) (Bendat & Piersol) — that "honest gap" is
closed; the two that remain open are per-shot input standardization and the physical
identity of the 50 ms saturation. **UF-CES exists as KSTAR hardware since 2021** (Jang,
Ko et al., RSI 92 053525: 10 µs Ti/rotation fluctuations, C VI 529.05 nm) — none of the
twelve's papers use it, but the acquisition-email question "UF-CES 취득분 있으면" is
grounded in real hardware; if such data ever arrives it is a NEW label channel and needs
its own pre-registration (PREREGISTRATION_B6.md §5's stance).

## Measuring On Rented Hardware (2026-08-23) — three lessons that cost real money

The IonQ branch (`docs/ionq_qpu_실험기록.md` §8–14) reached hardware for the first time. The
physics verdict lives there; these are the lessons that generalize to any paid, queued,
externally-metered measurement.

**1. One observable near a noise fixed point tells you nothing about noise.** The deviation of
the circuit's `<Z_0>` from its noiseless value came out *smaller* than the shot-noise
prediction (χ²/dof ≈ 0.06 on the first four points), which reads as "the QPU is
suspiciously good". Total variation distance over the full 256-state distribution said the
opposite: hardware sits **1.5–2.0× further from ideal than a perfect sampler**, and the ratio
grows with shots — a fixed error floor emerging as sampling noise shrinks. Depolarizing-type
error pulls a distribution toward uniform, and uniform has `<Z_0>` **exactly 0**; our operating
points already sit near 0 (1σ = 0.116), so error had almost nothing to move. Judge hardware at
the distribution level, never from a single scalar whose ideal value sits at the noise fixed
point.

**2. A range measured on a handful of points is not a range.** The circuit's output window was
first written up as 54.7 eV from **three** samples. Over 1,500 val points it is **361 eV**
(p1–p99) — 7× larger. That flipped the conclusion: shot noise at 400 shots is 9% of the window,
not 57%, so sampling precision was never the binding constraint and the VQC's loss is
expressivity, not measurement. Any "span", "range", or "window" quoted for a paper must come
from the population, and the analysis script now computes it that way rather than from whatever
points happened to get measured. See also [Numbers Must Come From Artifacts].

**3. Bill-by-job means a killed process burns money.** Two interruptions cost real charges
because completed QPU jobs whose results never reached disk were simply re-bought, and one
rerun truncated four already-paid records by reopening the result file. Fixes now in
`ionq_hw_ladder.py`, and the pattern applies to any metered external service:

- Name remote work by its **input identity** (`v1690_400sh`), never by position in the current
  plan (`s03_400sh`) — a position is meaningless once the plan changes, and re-deriving the
  mapping afterwards is manual archaeology.
- **Reclaim before you buy**: on startup, scan the provider's job history for completed work
  matching the plan and fold it into the resume set.
- Write results **after every unit**, and treat the result file as append-only state, not as
  something a rerun may truncate.
- Reconcile against the provider's own ledger, not your own tally. `quotaUsage` is the truth;
  ours disagreed by $51.58 until the stranded jobs were recovered.

**4. Measure the price, do not assume it.** The billing model was assumed to be per-gate-shot
from published rate cards; it is actually `2QGE_operations` with a per-job **minimum** that
dominates for small circuits, producing a flat two-tier fee — $25.79 up to 400 shots, $168.20
from 500 (debiasing forced on, not disableable) — **independent of circuit size**. Free
`dry_run: true` submissions return an exact quote and charge nothing, so there was never a
reason to guess. A 7× budget difference hung on one free API call.

## The Loss Is A Tail Statistic — And That Is Why Capacity Never Helped (2026-09-03) — §8aq

Three separate axes had already reported "a bigger estimator does not help": the 21k rung matching
the backbone (§8z), the flat 26× width sweep (§8aa), the family tie within 0.023 (§8ag). All three
were **relative** ceilings. Measuring the target's own reproducibility supplies the missing absolute
one, and it changes what those three results mean.

**The measurement.** Difference-based noise estimators (Rice / GSJS / order 3 / order 4 + a robust
MAD variant) over the full 641-file census, under the confirmed data treatment. Cut-population
`CES_TI`: order-4 bound **129.9 eV**, bulk (MAD) **46.4 eV**, against a backbone RMSE of 157.8 eV.

**The lesson, and it is a correction.** The two numbers disagree because **46.6% of the squared
mass sits in the top 1% of one-step changes** (65.6% for `V_rot`). So:

- In the **bulk** the model is 2.3–3.4× above the target's own scatter — there *is* headroom, and
  "the ceiling is information, not the estimator" **overclaims** if stated unqualified.
- In the **aggregate** the tail dominates, which is exactly why width and family were flat: they
  were being scored on a statistic that 1% of the rows decides.

**How to apply.** (1) Never write the ceiling claim without the qualifier "for the aggregate
MSE-based loss". (2) When a capacity or architecture axis comes back flat, check whether the metric
is tail-dominated *before* concluding the information is exhausted. (3) The named repair is a robust
co-metric (median absolute error / trimmed / Huber skill) re-scored from the frozen artifacts — no
retraining — and because it touches TEST it is **pre-registered before it is run**. (4) `√γ(1) =
151.8 eV` is within 4% of the backbone RMSE and `γ(1) > γ(2)`: the 10 ms grid is nearly pure nugget,
so do **not** quote a variogram-fitted nugget — the fit is not identifiable there.

See also [Numbers Must Come From Artifacts] and the §8ab spike-anchor audit, which is the same fact
seen from the `V_rot` side.

## A Conservation Law Closes A Negative Better Than An Ablation Does (2026-09-03) - 8ar

승상님's read was that the work is over-invested in validation and under-invested in the
mathematical and physical grounding of the model itself. It is measurable: `main_ko.tex` runs 53
lines of model against 596 of results, with nine tables, seven figures and **zero displayed
equations**.

**What fixed part of it, cheaply.** Writing the two transport equations down changes the question
being asked. They constrain a *rate*, not a level, so the right question about rotation is not
"do the fast diagnostics correlate with `V_rot`" - S8ab settled that with an ablation - but "do
they correlate with `dV_rot/dt`". Asking it term by term turned an open assumption into a closed
audit:

| term | disposition |
|---|---|
| LHS `dL/dt` | measured (consecutive observed pairs) |
| `T_NBI` | absent from the dataset (S8b.3) |
| `T_NTV ~ dB^2` | closed negative already (S8b.2: `|MC|`, rolling RMS) |
| `div(Pi_turb)` | **the one untested term** - measured null here (BES to `dV_rot/dt` r = -0.006) |
| `T_intrinsic` | needs `grad T_i`; we hold a scalar |

The internal control is what makes the null usable: the same measurement finds `+0.070 / +0.078`
for `dT_i/dt`, so the method can see a rate coupling when one exists.

**How to apply.** (1) Before running another architecture or ablation against a stubborn target,
write the governing equation and check which terms your inputs can carry - it is cheaper than a
batch and the negative it produces is stronger. (2) Report an empirical threshold with a physical
scale beside it: the ~50 ms context saturation now sits next to `tau_eq` = 8-59 ms at CES-relevant
`(n_e, T_e)`, as an order-of-magnitude consistency and nothing more. (3) Derive first, cite the
statistic as confirmation: the Mirnov loss is the sampling theorem (10 ms grid, 50 Hz Nyquist, kHz
modes), and S8b.2's lag-1 -0.009 is its consequence.

**And one new acquisition item.** `V_rot`'s own relaxation scale is **unmeasurable** in this
dataset - held-free gives 16 ms, held-kept gives >300 ms, a factor of 19 - because 54% of its
observations are instrument repeats. `T_i` is clean (1 held row in 641 files) at 159 ms, matching
BES 161 and ECEI 147. The held pathology costs not only 54% of the values but the ability to
characterize the quantity at all. See also [Stated criteria must be investigated].

## Estimate The Optimizer Exactly, Do Not Sample It (2026-09-03) - 8as

The batch size was inherited, never derived, and closing that gap took two attempts.

**The first attempt is the lesson.** McCandlish's two-batch-size estimator gets `|G|^2` by
subtracting two sampled gradient norms. Once the model converges and `|G|^2` is small those
two numbers are nearly equal, so the difference is all sampling noise: it returned a
**negative** `|G|^2` at two of thirteen checkpoints and a `B_simple` spanning 0.8-110 blocks.
More draws move the failure point; they do not remove it.

**The fix was to stop sampling.** With the block as the sampling unit - which is what the
training loop draws - `G = mean_i g_i` and `tr(Sigma) = mean_i |g_i|^2 - |G|^2` are computed
exactly in one pass, holding only a running gradient sum and a running scalar, and
Cauchy-Schwarz keeps `tr(Sigma) >= 0` by construction. Cost is one batch-1 backward per block
per checkpoint.

**Result.** `B_simple` post-init: median 33.6 blocks, geometric mean 28.4, range 6.0-61.4.
Batch 16 sits at about half the noise scale, so it is inside the regime where batch and step
count trade linearly - not wasteful, not obviously suboptimal, and doubling it would roughly
halve the step count.

**How to apply.** (1) When an estimator needs a difference of two noisy quantities, check
whether the exact version is affordable before adding draws - here the exact computation was
*cheaper* than the failed sampling. (2) Report the stable ratio, not just the headline one:
`mean|g|^2` and `tr(Sigma)` fall smoothly while `|G|^2` swings tenfold, so `B_simple` is an
order-of-magnitude statement and the durable finding is that **86-98% of a single block's
gradient energy is noise**. (3) Measure with dropout off and say so - `B_simple` is about
example-sampling noise, and the training loop's own scale is then a lower bound. (4) `muP`
is still unrun; until it is, 8aa's flat width sweep cannot fully separate "capacity does not
help" from "the learning rate did not match that width".

## The Closest Prior Work Is On Our Own Machine (2026-09-05) — §8at

**NN-CES** (Lee et al., *Fusion Eng. Des.* 222:115518, 2025; KFE + UST + UNIST, W.H. Ko among the
authors) trains a physics-constrained network on ~150,000 KSTAR frames and reads `T_i` and rotation
directly from the CES **spectra**, with no beam modulation. It is the nearest published neighbour to
this thesis, and the first literature sweep missed it.

**Why it was missed, and the fix.** The sweep asked for "prediction", "virtual diagnostic" and
"super-resolution". Fusion files this work under **analysis, inference, profile fitting**. Five more
papers came with it once the vocabulary changed: SVMR+GPR `T_i` profiles on KSTAR (NF 2024),
outlier-robust Bayesian mtanh fitting on KSTAR (NF 2026 — built so the CES term can be swapped for
NN-CES output), TSER recovery of missing electron temperature on EAST (NF 2025), full-state
reconstruction from sparse probes (arXiv 2026), and the DIII-D/MAST equilibrium challenge (arXiv
2026). **Search fusion literature through the OpenAlex API, not a web search engine**, and put
`analysis`, `inference`, `profile fitting` and `reconstruction` in the query.

**The positioning sentence, fixed.** NN-CES reads the instants where CES measured; this work fills
the instants where it did not. Its input is the CES spectrum itself, so it cannot run on a frame that
does not exist. Say it that way — it concedes the neighbour and keeps the distinction.

**And the discipline that followed.** The paper's "no modulation needed" invited an inference about
*our* missingness, and it was written into the appendix before it was measured. The measurement
(§8at) refused it: the two target masks agree only **72.8%** raw — our 3 keV cut moves that by 0.3
points — and the mask autocorrelations decay smoothly with no periodic ripple, so there is **no
shared shutter and no duty cycle** in the delivered 10 ms product. What survives is the *useful* half:
per-frame fit quality would replace our value-based cut and merge the two co-primary populations.
Same rule as always — a mechanism claim gets measured before it is published, even when the mechanism
comes from a peer-reviewed paper about our own instrument.

## The Headline Skill Is A Tail Effect — Read Every MSE Skill Beside Its Median (2026-10-12) — §8ax

B.13 re-scored the frozen TEST artifacts under robust co-metrics, with the rules frozen first
(`experiments/PREREGISTRATION_B13.md`). The MSE skill the thesis quotes (`T_i` +0.24 cut / +0.28
incl vs PCHIP) is earned **above the 70th percentile of error**. In the median row the backbone is
*worse* than PCHIP (median AE 47.2 vs 39.4 eV; `V_rot` 4.3 vs 2.9 km/s) and sits **with
persistence** (46.1 eV) — i.e. at the target's own bulk scatter (§8aq MAD 46.4 eV). The bulk
ordering is the ordering of noise averaging: acausal GP ≈ PCHIP (two neighbours) < causal GP
(several past samples) < persistence ≈ **backbone** (one noisy anchor, carried). `V_rot`'s n.s. is
not a tail artifact: it is a −49 % median loss offset by a tail win (peak-stratum MSE +0.5…+0.8).

**How to apply.**
- Never quote an MSE skill without the physical-unit median beside it. The paper's claim sentence
  is now "`T_i` 우위는 꼬리에서 온다" (pre-registered wording, §8ax §3), and the PR4 headline stays MSE.
- "Bulk headroom" (§8aq) now has a mechanism: an MSE loss dominated by rows the model cannot fit
  teaches nothing about the rows it could, so the model carries its anchor. The interventions that
  target this are the loss (B.14, running) and anchor denoising / shrinkage — not capacity (§8aa).
- The inclusive population is worse in the **bulk** too (62 vs 47 eV median): the two-population
  rule protects the typical row, not only the spike rows.
- A tail-driven result is not a weaker result — the model's value is in transients (§8an) — but it
  is a *different* result, and the deployment story must say "use the nowcaster where the quantity
  moves; in the quiet bulk an interpolator or the causal GP is better".
- Tooling: `experiments/b13_robust/rescore_b13.py --extra name=dir` scores any new arm the same way;
  the weighted-median cluster bootstrap is the pattern for any future quantile statistic.

## The Objective Moved The Median When Capacity Never Did — And The Checkpoint Rule Must Match The Loss (2026-10-12) — §8ay

**What happened.** B.14 changed one thing — the loss's influence function, with the Huber /
Student-t scale pinned to the target's own 10 ms reproducibility (§8aq) — and the typical
`T_i` row improved 8 % (47.2 → 43.4 eV, M1 +0.081, 4/4 splits) and `V_rot` 10 %, at zero
MSE cost in the cut population. Three capacity axes (§8z, §8aa, §8ag) had moved nothing.
The reason is §8ax: an MSE loss that 1 % of the rows decides teaches nothing about the other
99 %, so the bulk headroom was an *objective* problem, not a *model* problem.

**The trap.** In the inclusive population the same arms came out *worse* on MSE, and the
log said why before any analysis did: 7 of 8 runs stopped at epoch 8–10 with the best
**val MSE** at epoch 2–4. A loss that ignores residuals beyond 274 eV has no reason to
lower a validation MSE that those residuals dominate (85 % of the inclusive squared mass
in 1 % of rows), so the MSE-based checkpoint rule — kept so that all arms were *selected*
identically — picked a nearly untrained model. The arm's median still improved (+0.157).

**Rules.**
1. When an arm changes the training objective, the checkpoint-selection criterion is a
   *second* variable whether you like it or not. Either select on the arm's own objective
   (and say so), or fix the epoch count, or report both. Selecting a robust arm on val MSE
   is not "fair", it is a handicap in exactly the population where the tail dominates.
2. Read a robust arm's MSE deficit beside its epoch trace. A flat trace at epoch 2–4 is a
   selection failure, not a loss failure, and the record must say which.
3. Redescending influence (Student-t, ν = 3) gave up useful tail signal: tnll was behind
   huber on every metric in both populations. The scale-pinned Huber is the arm to carry.
4. The deployment statement now has two numbers, by design: aggregate skill (MSE, the tail)
   and median error (the bulk). Huber keeps the first and buys the second; a result that
   quotes one without the other is reading only one half of §8ax.

## The Model-Family Axis Closes Like The Estimator Axis Did — Context Carries The Shot, Not The Encoder (2026-10-12) — §8az

**What happened.** Three inductive biases (shot-latent neural process, MAST-pretrained TokaMind
transformer, the same transformer from scratch) were carded against a bidirectional LSTM control
on the offline gap-filling benchmark, 32 GPU runs, two populations. `V_rot` tied for the family;
the neural process was *worst* in the one regime it was built for (zero same-shot labels, where
no arm beats the training mean); the pretrained and random transformer backbones agreed to
0.004–0.013 on every cell.

**Rules.**
1. **Cross-device pretraining transfers nothing to CES gap-filling at this scale.** Do not
   re-try a foundation-model transplant without a new reason; the controlled pair
   (`tm_pre` vs `tm_scratch`) is the cleanest null in the project and matches TokaMind's own
   −0.005.
2. **The shot's regime is carried by its context labels, not by an encoder.** k = 3 → 10 labels
   lifts every family by the same +0.25; k = 0 sinks every family to the mean. A shot-level
   prior that beats this must be *conditioned on something the diagnostics do not contain*
   (beam sources, `I_p`, `B_T`) — that is the named next measurement, not another encoder.
3. **Read gap length before reading the verdict.** Every learned arm loses to PCHIP in one-step
   gaps and wins +0.4–0.6 beyond; the causal 10 ms `V_rot` n.s. and the offline 100 ms `V_rot`
   win are the same model seen at two scales. State which scale a `V_rot` claim is about.
4. **The val-MSE early-stop trap of §8ay is not loss-specific**: an `np` run stopped at epoch 3
   and lost 0.12 on one split. Any arm whose training signal is not the selection metric needs
   its epoch trace read before its number is believed.

## Useful Reference

**Dataset provenance and geometry (provider's note, relayed by 승상님 2026-10-26).** The 641 shot
CSVs are `/home/huiwon/KSTAR/data_set_2022/s{shot}.csv` on the lab server: discharges 30801–32751
(the 2022 campaign) minus those with insufficient data. The provider says to **ignore the first
and last row of every file** — measured here: those rows sit at t = 0.0 s and t ≈ 30 s, 11–30 s
away from the data block, carry no CES label in any of the 641 files and have complete inputs, so
they never become a scored sample or a sequence block (`build_blocks` drops 1-row blocks); their
only footprint is 2 of ~500 rows in each shot's per-shot input statistics, which is noise-level.
Recorded as a known deviation, not fixed, because dropping them changes cache signatures and the
frozen pairing. **`CES_TI18` / `CES_VT18` are the 2022 CES channel at R = 2.198 m** (the
labelled chord; our `CES_TI` / `CES_VT`). Fast-channel positions (approximate — exact positions
need per-time-step `syndia` equilibrium correction, which the provider considers impractical at
this volume, so "nearby" is the intended use): BES `0105–0107` at R ≈ 2.184 / 2.195 / 2.207 m,
z ≈ −0.011 m; `0205–0207` at R ≈ 2.182 / 2.194 / 2.205 m, z ≈ +0.002 m; `0305–0307` at R ≈ 2.181 /
2.192 / 2.203 m, z ≈ +0.017 m; ECEI `GT1103 / 1203 / 1303 / 1403` at R ≈ 2.193 m, z ≈ −0.025 /
−0.008 / +0.008 / +0.025 m — i.e. **every fast channel is within ~1.5 cm of the CES chord**, which
is the geometric basis for the T_i transient chain (same flux-surface neighbourhood). The Mirnov
coils are two positions (3 and 16) of a **poloidal array at one toroidal location**, so the two-coil
cross-phase measures poloidal wavenumber, not toroidal rotation — part of why the §8bd "mode
rotation" feature was never a rotation readout. Open question for the provider: the note names
`MC1T06`, the CSV header says `MC1T16`.

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

## Derive, Then Check Each Derivation Against The Frozen Record (2026-10-13) — §8ba

`docs/theory_foundations.md` derives the measurement physics (CES Doppler line, Cramér–Rao,
photon budget), the transport physics (equipartition, torque, force balance, Nyquist), the
estimation theory (filtering ⇔ recurrence, sufficient statistics of the carried/staleness features,
masked-loss minimizer, held shrinkage, noisy-OU closed forms for every baseline) and the evaluation
statistics (skill dilution, quantile integral, design effect, conformal, Huber influence, ensemble
identity). `ces_prediction/analyze_foundations.py` checks each one.

**Lessons that are new, not restated.**
- **Fit a noisy OU to the target before arguing about baselines.** Three numbers (σ_n, σ_s, τ) from
  the robust semivariogram predict the whole one-step ladder to within 10 % and give a *bound* for the
  deployable causal median (0.875× persistence). Any arm's bulk behaviour can be read against it:
  the MSE backbone 1.03×, Huber 0.94×, causal GP 0.96×.
- **The causal GP is a weak optimum.** The one-observation optimal shrinkage alone beats it in the
  median. If a "strongest deployable baseline" is needed, use an exact OU/Kalman arm.
- **Two derived scales now sit beside 50 ms**: `τ_eq` (8–59 ms) and the noisy-Markov filter memory
  (1–2 steps, saturating at 5). Quote both, each as order-of-magnitude.
- **Every evaluation device is an identity on the npz.** Spike dilution, quantile-weighted skill,
  ensemble disagreement, cluster design effect — check them with
  `analyze_foundations.py` Part E before interpreting a new arm's number.
- **How to apply.** When a result looks empirical and unanchored, write the governing equation or
  the generative model first (§8ar's rule), then compute what it *predicts* and compare; the
  comparison either confirms the mechanism or names the missing key (here: a future-anchor-distance
  key and an OU causal arm).

**Addendum (same day) — the plasma-behaviour physics (`docs/theory_plasma_physics.md`, Part F).**
- **Pr ~ 1 is readable from the CES series themselves**: 2SLS relaxation times `T_i` 103–140 ms,
  `V_rot` 89 ms, IPB98 τ_E 77 ms. Use 2SLS (instrument `y_{t−1}`), never OLS, on a noisy series —
  OLS returns 32–36 ms, which is the noise attenuation bias, not physics.
- **The non-CES route to `V_rot` is the mode Doppler** `v_φ = 2πR f / n` (EHO 4 / 8 kHz → 53 km/s vs
  CES median 69). It is the physical content of B.6's "mode rotation frequency" feature; frame it as a
  measurement of rotation, not as a correlate.
- **The physics-based `T_i` predictor is blocked by calibration, not by physics**: the coupling term's
  sign is random across blocks when the driver is the uncalibrated ECEI mean (46 % positive). The
  acquisition item is one calibrated `T_e` on the CES flux surface.
- Quote the assumed plasma parameters (document appendix A) whenever a Part F number is used; none of
  them is in the CSVs.

## Explain A Result By Deriving It In Time t, Then Check The Derivation On The Frozen Rows (2026-10-13) — §8bb

Ten parallel proof agents each took one thesis result, derived it as explicit equations in time
(ODE solutions, impulse responses, convolution integrals, decay laws) from the plasma's
conservation laws, and checked the derivation on the 641-file census and the frozen TEST npz.
Synthesis: `docs/theory_results_physics.md` (one generative model M1–M7, ten proofs, relevance).

**Lessons.**
- **One equation, two coefficient sets.** `dy/dt = −(y−m)/τ + noise + Σ A_k ḣ(t−t_k)` describes
  both targets; `T_i`'s `m` and `A_k` are set by `(T_e, n_e)` the fast diagnostics see at the same
  instant, `V_rot`'s by torque nothing sees. Every headline follows: write the paired MSE
  difference as `D = p_obs r E_imp − c_b` (P08) or the skill as `f_I r_I − (1−f_I) ℓ_B` (P01)
  before arguing about architecture.
- **The `T_i` headline is 90–114 % transient term** (P01). Quote it with its decomposition:
  the model loses to the acausal GP in the quiet half of rows on 4/4 splits and wins on the top
  5 % jump rows on 4/4.
- **Condition on the innovation, not on level or rate** (P06). The fast-diagnostic bound is
  `Δskill = r²_{yz·h} MSE_h/MSE_pers`; the history-conditional partial r is 0.26, the §8ar rate
  correlation 0.07 is not the input. Measured +0.048 sits inside the linear bound +0.047–0.056:
  the backbone extracts what a linear reader extracts, no more.
- **Two corrections to the ledger, both annotated in place.** §8ab's "43 % PR2 fallback" counts
  unscored rows (`eval_seq.py` `n_total`); the scored-row fallback is 0.3–0.5 %. §8az's "`V_rot`
  recoverable from the fast channels at 100 ms" cannot be: the `bilstm` control's `V_rot` branch
  never sees them; it is the two-sided label context.
- **Campaign drift is `x = g(D) s(t) + c(D)`** (P04): between-shot variance share 0.84/0.91 for
  BES/ECEI vs 0.49/0.28 for the targets; per-shot z-score cancels `g, c` exactly; the online
  EWMA version has bias `−b τ_w` (measured 0.29–0.33 σ at τ_w = 1 s).
- **Missingness is photon-budget physics** (P07): missing rate rises 5.8× across target-jump
  quintiles, 19.9 % vs 3.7 % across BES level quintiles, and relaxes after an event with
  τ_B ≈ 26–35 ms — write MNAR as `P(obs | N(t), B(t))`, not as a label.
- **Real-time classes come from the plasma** (P10): the nowcaster serves transport control
  (class A) only; ELM triggering and mode control are excluded by the 10 ms label integration,
  not by inference latency, so the 1 ms verdict does not change the class.
- Ask a sub-agent for a proof the way these were asked: one result, one derivation ending in a
  number, one check on data, a "limits / what would overturn this" section, and deliver by file
  (message relays truncate at ~8 kB).

## The Coil Is Right And The Feature Is Wrong — The Mirnov Route Is Closed At Every Rate (2026-10-26) — §8bd

The μs raw data (12 frozen discharges, 36 CSV, 7.4 GB, `data/raw_mus/`) arrived and B.6 ran the
same day under the 2026-08-21 pre-registration. H1 — restored kHz Mirnov features into the `V_rot`
branch — is **rejected in both populations**: paired `V_rot` −0.128 (cut) / −0.164 (inclusive),
0/3 test discharges improved, 4/4 and 3/4 seeds significantly negative, block bootstrap agreeing.

- **The positive controls are the lasting part.** The 2 MHz Mirnov stream shows the published
  #32092 EHO at 4.39 / 8.91 kHz with prominence 10⁴, the extractor's mode channel reads it as
  100 % valid at 4 kHz, and **the 100 Hz MC values are bit-exact raw samples on 24/24 coils** —
  the "random-phase snapshot" premise (§8b.3) is now measured, not inferred. The BES controls
  failed at the registered windows (the 3–6 s WCM window had been transcribed from the 100 Hz
  file, not the paper; the 100–300 kHz band is a noise floor on the delivered channels) — record
  a control's *source* (figure vs. our own file) when you register it.
- **Why a physically right chain still fails as a feature**: the coil's dominant 10 ms frequency
  is not a row-by-row function of CES edge rotation here — within-discharge Spearman |ρ| ≤ 0.11
  in 6 of 7 discharges (only the ERMP shot #31359 reaches +0.51). It jumps between modes and
  carries mode number as much as flow. Fifteen such per-shot-z-scored channels on six training
  discharges are a high-variance input that fits the pool and does not transfer; the loss sits
  in the middle/active `V_rot` terciles, where H1 expected the gain. Six one-shot validation
  folds did not foresee it (A1 within ±0.07 of A0 on val) — at this scale val is not a guard.
- **Delivery lessons**: no metadata came, so rates are measured (BES/ECEI 500 kHz, #31747 BES
  1 MHz, MC 2 MHz) and verified sample-by-sample; timestamps are quantised (1 / 10 µs) and must
  be reconstructed; #31873's MC is 2 s of 7.67 s; #31902's raw window misses a third of its
  `T_i` rows. A ratio test of a second moment on a kurtosis-15 signal needs thousands of
  windows, not hundreds — prefer an identity test when one exists.
- **How to apply:** the Mirnov route to `V_rot` is closed at 100 Hz snapshots, 100 Hz derived
  envelopes and the raw 2 MHz stream through the registered feature set; a different raw-feature
  family (multi-coil cross-phase slope → v⊥) is a new pre-registration, not a re-tune. Never quote
  A0's +0.18 / +0.21 `V_rot` skill on the three test discharges as a `V_rot` result — six training
  shots, three test shots, context only. (The "NBI torque channel is the remaining lever" sentence
  that stood here was corrected on 2026-10-29 — next section.)

## A Step-Held Driver Cannot Move A 10 ms Increment — The `V_rot` Result Is Final, Not Provisional (2026-10-29) — §8bd addendum

Every `V_rot` record since §8b.3 ended with "the torque driver (NBI) is unobserved", read as a
provisional result waiting for a channel. 승상님 asked the question that closes it: KSTAR NBI power
is **piecewise constant** (ion sources switched at a few programmed times, held at fixed voltage
and current in between; the literature's modulation experiments are the exception), so what would
the channel add at 10 ms?

- **Measured on our own labels**: the CES beam was on continuously in the 2022 discharges (label
  fill 94 % of the 10 ms grid, longest unbroken run 1.3 s median), and the > 300 ms band carries
  **0.3 %** of the 10 ms `V_rot` increment variance (IQR 0.2–0.9 %, 20 shots; `T_i` 0.25 %). A
  step-held input can act only in that band, so a perfect NBI channel bounds the removable 10 ms
  increment variance at ~1 %. The one published trace we could overlay (#31359, Kim et al. 2024
  Fig. 6a) shows it: the single 3.5 → 4.0 MW step moves the `V_rot` level by ~40 km/s within
  0.2–0.7 s, a later 150 → 60 km/s drop happens with the heating constant (ETB formation), and the
  10 ms increments are the same width before, after and through every beam notch.
- **The two-scale sentence for the thesis**: the *macro level* of `V_rot` is set by the torque
  balance (co-NBI + intrinsic torques over density and momentum loss; Shi 2016, Yang 2018) and is
  already in the nowcaster's input as the carried observation; the *10 ms increment* is intrinsic
  torque, ELM momentum loss and noise, which no 0D channel carries. So P02's "history-only OU
  filter" is the result *with or without* the driver observed. The NBI data request is deferred;
  it matters only for ≥ 100 ms forecasting after power steps or for the cold start.
- **How to apply**: before calling an unobserved driver "the lever", measure how much of the
  target's increment variance sits in the band that driver can occupy. A driver that is constant
  on the prediction horizon explains levels, not increments. Do not run a transcribed-trace pilot
  to "check": with a 0.3 % bound and one discharge its CI cannot show anything either way.

## The Nowcast Error Is Heavy-Tailed, Not Two-Valued — A Scale Mixture Is Calibrated Where The Gaussian Is Not (2026-10-29) — §8bf

B.17 put a predictive distribution on the unchanged seq_v2 (NLL-trained Gaussian `v2g` vs
two-component mixture `v2m2`) expecting, from §8bb, a *two-valued* conditional at pedestal events.
The data said otherwise, and the registered rules, bound to the Gaussian arm and to a bimodality
test, could not name the arm that worked — the result is exploratory by the letter and the
strongest distributional finding on record by the numbers.

- **What was found**: `v2m2` beats the oracle-scale homoscedastic Gaussian by +0.215 / +0.438
  CRPS (`T_i`, cut / incl) and +0.34 / +0.35 (`V_rot`), 16/16 splits significant; it is calibrated
  on `T_i` (50 % / 90 % coverage 0.51–0.56 / 0.89–0.92); and its posterior mean beats the
  MSE-trained backbone on cut `T_i` 4/4 (pooled +0.028) while tying on `V_rot`. The mixture fires
  the bimodality test on **zero** rows: its second component is wide (σ₂ 90th pct 545 eV) with
  7–10 % weight — a **scale mixture**, i.e. a heavy-tailed one-peak error. The Gaussian head covers
  62–80 % of rows with its central 50 % interval (σ inflated by tails it cannot shape) and is
  rejected on `V_rot` MSE (3/4 significant losses): the 1/σ² NLL weight collapses on the
  |v| > 1000 km/s rows and drags the mean, which the mixture parks in its wide component.
- **The physics readouts**: `T_i` σ rises 1.2× on event rows and ρ(σ, event energy) ≈ +0.17 — the
  §8bb p_obs signal at the predicted size, not more; `V_rot` σ has **no** staleness dependence
  (ρ ≈ 0.00), so the learned uncertainty is a per-shot scale, not the OU variance growth the
  Kalman cell encodes — a second sense in which the `V_rot` path is history-only.
- **How to apply**: this is the same tail fact as §8ax/§8ay seen from the likelihood side — the
  mixture NLL is the robust loss B.14 was looking for, and it buys calibration for free. Report
  `v2m2` as the calibrated arm under the exploratory label; the confirmatory registration is a
  Student-t head (`v2t`, learnable ν) against `v2m2`, with the claim bound to a named arm *before*
  numbers. Registration lesson: never tie a claim to one arm and a mechanism test to another —
  write the claim over "the arm that passes the guards", with the guards fixed in advance. Seven
  of 16 runs hit the 60-epoch cap (`v2m2` 6/8); treat the mixture's numbers as a lower bound and
  register a higher cap next time.
- **Confirmed by B.17b (2026-10-30, §8bj)**: with the claim bound to "the arm that passes four fixed
  guards", it is supported in both populations. Cut: Student-t (`v2t`, ν ≈ 3) ties the mixture and
  is named (fewer parameters) — the one-peak heavy tail is now measured. Inclusive: only the
  mixture passes; the t head's posterior mean loses −0.13 MSE on spike rows because the t location
  is redescending and ignores > 3 keV fit failures that MSE still scores. Lesson: a redescending
  likelihood's mean is a robust location, not a conditional mean — wherever the scored population
  keeps outliers, use a scale mixture (or report the median, not the MSE). `V_rot` ν sits on the
  2.05 floor: its increments are jump-like, a third sign after §8bf and §8bi. Cap 100 vs 60 moved
  the mixture only +0.01.

## The Structure The Derivation Asked For Buys A Third Of What It Promised — And The Gate Is Negative (2026-10-29) — §8bg

B.16 built the output the derivation audit located (§8ba/§8bb): a learnable OU–Kalman anchor with
the seq_v2 LSTM as residual and an event gate on the fast-channel increment energy. The anchor is
real: the `T_i` median crosses below persistence for the first time (cut 1.024× → 0.978×,
inclusive 1.337× → 1.089× where K < 1 shrinks the spike anchors exactly as §3 predicted), the
quiet-discharge win rate vs the causal GP reaches the predicted 0.50, aggregate skill is a tie,
and the cell's τ / σ stay at their physical values. It also lands at 0.98×, not the 0.875–0.94×
promised, so the structural route recovers about 30 % of the headroom and less than the Huber loss
did (§8ay). Rule 1 fails by 0.01 in the cut population and passes 4/4 in the inclusive one — an
inclusive-conditional rung, the mirror of B.5's cut-conditional one.

- **The gate is negative** (`v4k − v4kng` −0.008 cut, −0.045\* inclusive; quiet win 0.50 vs 0.54).
  The ungated residual already suppresses itself where it should; a gate initialised at σ(−2) =
  0.12 throttles the residual's gradient under patience-6 early stopping and leaves `v4k` as an
  under-trained `v4kng`. Candidate mechanism, one-line test: initialise the gate open.
- **Where the bulk headroom actually is**: rows with stale history (Δt > 15 ms) lose M1 by
  0.11–0.28 on 8/8 cells. The OU propagation relaxes the anchor toward the slow level at rate ρ
  while the truth, whose increments are bursty (§8ar), does not. The long-gap regime wants a
  level-holding prior, not an OU prior — that is the next structural registration, and it is one
  parameter.
- **`V_rot` is untouched** (median 1.16× vs 1.17×, aggregate slightly worse): the residual head
  re-introduces in the bulk what the anchor removes, and a gate that sees only the slow state has
  nothing to close on. Same fact as §8bf's "the `V_rot` uncertainty ignores staleness". Whether the
  P02 bound of 0.921× is reachable by the pure filter is a named inference-only readout (`r ≡ 0`
  on the `v4kng` checkpoints), registered before it is run.
- **How to apply**: a Kalman cell written as a per-step Python loop costs 10× the registered
  estimate (20–25 min per run); register cost from a smoke timing, not from the backbone's.
  When a registration defines adoption on one population and the other one passes, say so with the
  mirror label rather than "not adopted" alone, and fix the asymmetry in the next registration.

## The OU Prior Is Right For The Tail And Wrong For The Bulk — The P02 Median Bound Is An OU Number (2026-10-29 … 30) — §8bh, §8bi, §8bk

- **Checkpoint selection must match the objective** (§8bh): re-selecting the inclusive Huber arm on
  its own val loss let it train 30–60 epochs instead of 8–23 and doubled its median gain (`T_i` M1
  +0.285, `V_rot` +0.145, `V_rot` median 1.02× persistence — the first arm inside the P02 band),
  at a pooled `T_i` MSE cost of −0.019 with one significant split, so the rung stays
  cut-conditional by the letter. A robust loss early-stopped on val MSE is not a test of the loss.
- **Three B.16 follow-ups** (§8bi): an open-initialised gate learns to stay open and reproduces the
  ungated arm (the gate carries nothing); a level-holding anchor wins the 2.5 % stale rows by the
  predicted +0.08 / +0.10 and loses the other 97.5 %; the pure learned OU filter sits at 1.19×
  persistence in the `V_rot` median. Increments are bursty (hold, then jump), so the OU filter wins
  aggregate MSE and loses the median; the 0.921× bound of §8bb P02 holds only under the OU prior.
- **The hold-or-jump anchor was measured (B.18, §8bk)**: alone it fixes the `T_i` median (0.979× /
  1.001× persistence vs OU 1.060× / 1.115×; inclusive +0.25 M1 over the backbone) and loses aggregate
  MSE to PCHIP — the mirror image of OU. With the MSE-trained residual on top the median gain is
  spent (cut ties `v4kng`, inclusive −0.134), so it did not replace the OU anchor. The learned jump
  rate stays within 3 Hz of its 5.1 Hz init: end-to-end MSE does not identify it.
- **`V_rot` sits at 1.18× persistence under both level priors** (OU 1.189 / jump 1.181): the prior's
  form is not the cause. Either the median rows trend (a level prior lacks a slope term) or σ_n
  learned end-to-end shrinks real signal.
- **How to apply**: do not re-try level priors (OU variants or hold-or-jump) for `V_rot`. First measure
  the lag-1 autocorrelation of observed `V_rot` increments on TRAIN; positive → local-linear-trend
  cell, ≈ 0 → σ_n fitted by the anchor's own innovation likelihood. For `T_i`, the open row is a
  residual trained/selected on the median objective (§8bh rule) on top of the jump anchor.
