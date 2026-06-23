# Progress Log — CES nowcasting thesis (model vs conventional interpolation)

Running summary of changes/findings this work stream. Newest at top. (HANDOFF.md = per-run
auto-state; PROJECT_KNOWLEDGE.md = long-term lessons; this file = a brief human-readable changelog.)

## Thesis (crystallized via deep-interview → omc-plan consensus)
- **Claim**: the model (fast diagnostics BES/ECEI/MC + past CES history) beats conventional
  past+future CES interpolation (linear/PCHIP) for **CES_TI**; CES_VT non-win = T_i↔V_rot asymmetry.
- **Success gate (PR4)**: CES_TI beats best interpolation with 95% **shot-clustered bootstrap** CI
  excluding 0, on a **held-out test** split (selection on val only).
- Spec: `.omc/specs/deep-interview-ces-nowcasting-thesis.md`; Plan: `.omc/plans/ces-interpolation-comparison-consensus.md`.

## ✅ CONFIRMED RESULT (final model = AutoML-improved iter5: GRU + multi-head attention per-target heads)
Across **4 independent held-out test splits** (seeds 42/1/7/123; seeds 1/7/123 never touched
architecture selection), shot-clustered paired bootstrap:

| target | skill_vs_pchip range | all CIs exclude 0? | verdict |
|--------|----------------------|--------------------|---------|
| **CES_TI** | +0.20 … +0.30 | **yes (4/4 PASS)** | **model significantly beats PCHIP & linear interpolation** |
| CES_VT | +0.08 … +0.16 | no (4/4 n.s.) | not significant → T_i↔V_rot asymmetry |

**Thesis claim CONFIRMED (pre-registered success shape):** the model (fast diagnostics + past CES)
beats conventional past+future interpolation for **CES_TI** robustly; **CES_VT** does not — fast
diagnostics carry ion-temperature info beyond temporal CES interpolation, but no toroidal-rotation
info (NBI torque unobserved, Mirnov aliased). Also: the model beats all causal baselines
(persistence/AR) decisively.

### History of the result
- Earlier *baseline* model (iter2, GRU per-target heads) on its held-out split: TI +0.088, CI
  [−0.22,+0.32] → **n.s.** (the "keep improving" decision followed).
- The improvement loop (objective = val skill_vs_pchip) found iter5 (added multi-head attention),
  val skill_vs_pchip 0.179→0.207. Retrained + tested → the multi-seed PASS above.
- Caveat: baseline vs improved used different splits, so the n.s.→significant jump is not perfectly
  isolated to architecture; but the improved model is robust across 4 splits (incl. 3 fully
  independent), so the *claim itself* (model beats interpolation for T_i) is solid.
- Limitation retained: skill is on observed CES points only (MNAR optimistic bound); window=4.

## Built & verified (this session)
- `ces_prediction/baselines_interpolation.py` — linear/PCHIP/AR/(GP) over block neighbors; excludes
  target row (no leakage), ≥0.5s gap refusal, persistence fallback (PR2). **7 unit tests pass.**
- `ces_prediction/evaluate.py` — extracted shared `build_clean_val_subset()`; **byte-identical
  regression guard verified** (delete + regenerate).
- `ces_prediction/compare_baselines.py` — model-vs-interpolation, physical units, PCHIP headline,
  identical-sample, gap-stratified bins, per-shot squared errors saved for bootstrap.
- `ces_prediction/bootstrap_compare.py` — shot-clustered paired bootstrap CI (PR4 gate), aggregate
  + small-gap.
- `ces_prediction/train.py` — 3-way train/val/**test** split (held-out test) + `CES_INIT_SEED`
  (init decoupled from split). Backward-compatible (test_fraction=0 ⇒ identical 2-way). **pytest 10/10.**
- `ces_prediction/automl_agent_loop.py` — (a) `.env` auto-load; (b) researcher via **subscription**
  `claude` CLI; (c) search space = architecture family + `WINDOW_SIZE`; (d) **loop now optimizes
  `skill_vs_pchip` on val** (not just skill_vs_persistence), with a reserved held-out test split.
- Final model = restored AutoML best (`best_model.py`, iter2 GRU per-target heads, clean-skill 0.3717).

## Conventions / isolation
- All experiments use throwaway `data/.*` dirs; canonical split/weights preserved; `model.py` = best.
- scipy added (PCHIP); `python-dotenv`, `scipy` in `pyproject.toml`.

## Finalization (parallel team, non-GPU — ran while the improvement loop trains)
- `THESIS_RESULTS.md` — honest thesis-grade write-up (claim, method, held-out-test results, bootstrap
  null, causal-baseline superiority, limitations, framings). Numbers verified against the JSONs.
- `docs/interpolation_baselines_references.md` — annotated, web-verified bibliography for the
  interpolation baselines (Fritsch&Carlson 1980 PCHIP, Reinsch 1967 spline, Eyheramendy 2018 IAR,
  Chilenski 2015 / Michoski 2024 GP, Murphy 1988 skill score) + justification.
- `.omc/research/analysis_modules_review.md` + `tests/test_bootstrap_compare.py` (9 tests pass) —
  correctness review: **no HIGH findings**, all invariants PASS (no leakage, identical samples,
  persistence cross-check, shot-clustered bootstrap correct). Fixed MED-1 (guarded a 0-division in
  the bootstrap statistic). Deferred LOWs in compare_baselines.py until the loop frees the file.

## UPDATE — improved model beats interpolation for CES_TI (one held-out test; robustness pending)
The AutoML-improved model (iter5: GRU + **multi-head attention per-target heads**), retrained on its
own 3-way split and tested on its **held-out test** (selection was on val only):

| target | RMSE model | PCHIP | skill_vs_pchip | shot-clustered 95% CI | gate |
|--------|-----------|-------|----------------|------------------------|------|
| CES_TI | 368.9 | 431.8 | **+0.270** | **[+0.144, +0.364]** | ✅ PASS (beats) |
| CES_TI (Δt≤15ms) | — | — | +0.245 | [+0.125, +0.322] | ✅ PASS |
| CES_VT | 22.4 | 24.5 | +0.161 | [−0.382, +0.334] | n.s. |

This matches the pre-registered success shape: **CES_TI significantly beats offline interpolation;
CES_VT does not (the T_i↔V_rot asymmetry)**. CAVEAT: the earlier baseline (+0.088 n.s.) was on a
*different* split, so the baseline→improved jump is NOT clean attribution; and a single held-out test
could be split-luck. **Robustness check running** (multi-seed, below) before the result is declared.

## Runs
- **DONE** — `b9xbax3z2`: multi-seed robustness (seeds 1/7/123) → CES_TI PASS on all (+0.20/+0.27/+0.30,
  CIs exclude 0); CES_VT n.s. on all. Result is robust, not split-luck. (See CONFIRMED RESULT above.)
- **DONE** — improvement loop (10 iters, `bo7c2mxp6`). Objective = val `skill_vs_pchip`.
  Result: **val skill_vs_pchip 0.1792 (baseline) → 0.2075 (iter5, kept)**; iters 6–10 rolled back
  (stale 5). Best architecture = GRU history-encoder + **multi-head attention per-target heads**
  (`data/.improve_out/.automl_state/best_model.py`, window=4). So the researcher did find a better
  model on val (+0.028 skill_vs_pchip) — modest, as expected near the autocorrelation ceiling.
- **IN PROGRESS** — `brt40kunq`: retrain the improve-best architecture on its 3-way split
  (`data/.improve_split/w4`, test held out) for consistent weights, then `compare_baselines` +
  `bootstrap_compare` on the **held-out test** to check whether the +0.028 val gain makes the
  model significantly beat interpolation (honest expectation: still n.s. at ~96 shots).

## Next
- When the loop finishes: restore `model.py` from `data/.improve_out/.automl_state/best_model.py`,
  re-run `compare_baselines.py` + `bootstrap_compare.py` on the **held-out test**, and check whether
  the best model now beats interpolation with 95% CI excluding 0.
- Honest expectation: a significant win vs offline interpolation is hard at ~96 shots; the robust,
  true result is causal-baseline superiority + competitiveness with interpolation. If still n.s.,
  reframe (causal/online superiority) or accept the null + physics-asymmetry contribution.
