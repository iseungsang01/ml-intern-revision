# Research Plan — KSTAR CES nowcasting (thesis / scientific contribution)

North-star: **a defensible scientific contribution (thesis)**, not a model-architecture
leaderboard. The deliverable is a clear question, a sound method, evidence with error bars,
and honest limitations. Architecture of `model.py` is *infrastructure for the analysis*, not
the product.

## Core claim (what the thesis argues)

For KSTAR CES gap-filling/nowcasting on the ~10 ms grid:

1. **Target asymmetry (physics-grounded).** Toroidal rotation `CES_VT` is predictable
   essentially *only* from its own recent CES history; the fast core diagnostics
   (BES/ECEI/MC) carry ~no usable rotation information at this timescale (NBI torque is
   unobserved; Mirnov coils are sampled at 100 Hz, aliasing the kHz mode rotation). Ion
   temperature `CES_TI` is different: fast diagnostics *do* carry T_i information
   (collisional e–i coupling, t_ei ∝ T_e^{3/2}/n_e), so a model using them beats persistence
   by more than history alone would.
2. **Where learning helps.** A learned model beats the persistence baseline ("carry the last
   observed CES forward") as a function of the gap length Δt; we characterize that curve.
3. **Complexity is/ isn't justified.** A like-for-like comparison of the deep multimodal model
   vs history-only vs a simple linear/AR baseline shows how much network complexity the data
   actually supports.

## Primary metric (single, fixed)

`skill_vs_persistence = 1 − MSE_model / MSE_persistence`, **per target**, in **physical CES
units**, on the **clean (non-augmented) validation split**, evaluated only where the target is
observed AND persistence has a value (identical samples for both). Report with **error bars
across seeds**. Retire the augmented training `val_loss` as a progress metric — it is not a
clean generalization estimate and caused the historical metric incoherence.

> Known limitation to state plainly: CES missingness is **MNAR** (drop-out at low S/N, ELMs,
> transitions). Skill measured on *observed* points is an **optimistic bound** for the
> genuinely-missing points the model is meant to fill. This is unavoidable (no ground truth for
> missing points) and must be disclosed, not hidden.

## Workstreams (in priority order)

### WS1 — Gap-length stratified skill  ⭐ (highest priority)
- **Question:** Where (in Δt = time since last observed CES) does the model actually beat
  persistence? Is the plateau because samples are dominated by tiny gaps (~10 ms, where
  persistence is near-optimal due to autocorrelation)?
- **Method:** For each clean-val sample and target, compute Δt from the most-recent observed
  position using the `lookback_seconds` time feature (`time_features[:, last_idx, 0]`, raw
  seconds; handles irregular sampling). Bin by Δt and report per-bin `skill_vs_persistence`,
  RMSE_model, RMSE_persistence, and n. Script: `ces_prediction/analyze_gap.py` (does not touch
  the loop's `evaluate.py`).
- **Deliverable:** skill-vs-Δt curve per target (thesis figure) + the sample-count histogram
  over Δt (shows the easy-sample dominance).
- **Acceptance:** clear statement of the Δt regime where the model adds value, with n per bin.

### WS2 — Multi-seed modality ablation with CI
- **Question:** Is the T_i↔V_rot asymmetry statistically real, or seed noise (the current
  finding is single-seed, 40k, 12 epochs)?
- **Method:** Re-run the `CES_ABLATE` ∈ {none, no_fast, no_history} ablation across ≥5 seeds at
  a fixed budget; report mean ± CI of per-target skill. Reuse `train.py`/`evaluate.py` with
  `CES_ABLATE` and `CES_SEED`; throwaway split/output dirs per run.
- **Deliverable:** ablation table with error bars (thesis table) confirming/qualifying the
  asymmetry.
- **Acceptance:** the V_rot fast-only collapse and the T_i fast-only gain are outside CI.

### WS3 — Baseline hierarchy (is the deep net justified?)
- **Question:** Does the deep multimodal model beat simple baselines enough to justify it?
- **Method:** Compare, on the same clean-val protocol: (a) persistence, (b) per-target linear
  / small-MLP on history only, (c) history-only deep model, (d) full multimodal. Same metric,
  multi-seed.
- **Deliverable:** baseline-ladder table; conclusion on warranted complexity.
- **Acceptance:** explicit "deep ≈ simple" or "deep > simple by X (CI)" statement per target.

### WS4 — Documentation / write-up scaffolding
- Keep this file as the living plan. Fold confirmed results into `PROJECT_KNOWLEDGE.md`
  (long-term) and draft thesis-section bullets (claim → method → figure/table → limitation).
- Align `program.md`, `PROJECT_KNOWLEDGE.md`, and any loop scoring to the single primary metric
  above; stop tracking augmented val_loss as progress.

## Execution notes
- Run experiments from the **git-committed baseline `model.py`** (reproducible), not the
  uncommitted +0.02 iter-2 tweak, unless a workstream explicitly varies architecture.
- Real data only (641 CSVs in `data/`), GPU. Throwaway `CES_SPLIT_DIR`/`CES_OUTPUT_DIR` for
  ablation runs so canonical artifacts are preserved.
- Architecture search (the AutoML loop) is **paused** as a primary activity; it may return only
  as a controlled WS3 comparison.

## Status
- [ ] WS1 gap-length stratified skill (in progress)
- [ ] WS2 multi-seed ablation + CI
- [ ] WS3 baseline hierarchy
- [ ] WS4 write-up + metric/doc alignment
