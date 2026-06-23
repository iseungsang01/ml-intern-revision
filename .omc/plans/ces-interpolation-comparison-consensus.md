# Consensus Plan: Conventional-interpolation comparison harness for the KSTAR CES nowcasting thesis

Status: **pending approval** (omc-plan consensus, RALPLAN-DR short; Planner → Architect → Critic, 1 iteration, Critic APPROVED-with-conditions; all conditions folded in below)
Source spec: `.omc/specs/deep-interview-ces-nowcasting-thesis.md`
Execution: NOT started. A separate explicit execution approval is required before any code is written.

---

## RALPLAN-DR Summary

### Principles
1. **Identical-sample fairness** — model and every baseline scored on the same `(file, row_index)` set and the same per-target keep mask (extends `evaluate.py:201`).
2. **No leakage** — a baseline for target `t` may use past+future *neighbor* CES but never the target's own value at `row_index` (mirrors `dataset.py:381` masking).
3. **Read from `dataset.file_arrays[file_idx]`, never re-parse CSV** — `row_index` indexes the filtered array (`dataset.py:91-97`), not CSV lines.
4. **Reuse, don't fork, the clean-val protocol** — share one `build_clean_val_subset()` helper between `evaluate.py` and the new harness (kills drift).
5. **Honest statistics** — pre-registered baseline + test split + shot-clustered paired bootstrap; full baseline ladder reported; MNAR disclosed.

### Decision Drivers
1. Correct sample alignment + leakage exclusion (highest correctness risk).
2. A CI that actually supports "model beats interpolation" → resolve `train.py:274` seed coupling and use the right resampling unit (shot) and a held-out test set.
3. A defensible, literature-grounded, pre-registered baseline (no strawman, no winner's-curse).

### Options (key decision: how baselines access past+future CES and align with the clean-val protocol)
- **Option A (CHOSEN):** new `baselines_interpolation.py` + `compare_baselines.py` reading from `dataset.file_arrays[file_idx]` at `row_index`, sharing `evaluate.py`'s val-subset helper. Pros: guaranteed identical index space, zero risk to the AutoML scoring path, clean separation. Cons: a second harness — mitigated by the shared helper (Principle 4).
- **Option B (INVALIDATED):** re-parse CSV, join on `time`. `row_index` indexes the *filtered* array (`dataset.py:91-97`), so a CSV join misaligns / re-implements the filter. Kept only as an optional ~20-sample audit cross-check.
- **Option C (INVALIDATED as primary):** extend `evaluate.py` in place. `evaluate.py` is the AutoML keep/discard signal (`automl_agent_loop.py:259,286,343`); editing it risks perturbing the search. Its one virtue (drift-immunity) is recovered by the Option-A shared helper (Principle 4).

---

## Requirements Summary
Build evaluation-time **conventional interpolation baselines** (allowed past+future CES) and a **held-out-test, multi-seed, gap-stratified comparison harness** proving the fixed nowcasting model beats the best **pre-registered** interpolation baseline for **CES_TI** (and honestly reporting the **CES_VT** asymmetry), in physical units, with a statistically defensible CI. The data/model contract and the `evaluate.py`/AutoML scoring path are not modified (only an additive, off-path refactor + a new `CES_INIT_SEED`).

---

## Pre-registration (MUST be fixed in writing before viewing TEST numbers)
- **PR1 — Best interpolation rule:** primary headline compares the model against **PCHIP** (monotone cubic; the single named acausal method chosen for ELM/sawtooth robustness). The full ladder {persistence, linear, PCHIP, AR, GP-if-available} is also reported. If "model vs best-of-ladder" is reported, it is **Bonferroni-corrected over the acausal set** and the win must hold against PCHIP specifically.
- **PR2 — Headline evaluation population:** interpolation predicts at **every observed target the model is scored on**; where no future observed neighbor exists within the no-gap window, the interpolation arm **falls back to persistence (last observed)** so the arm is defined everywhere the model is. The keep mask is therefore the model's existing `mask[:,t] & persist_obs[:,t]` — NOT thinned by future-neighbor availability. The fraction of points using the future neighbor vs the fallback is reported per Δt bin.
- **PR3 — Held-out TEST split + floor:** a file-level TEST split is reserved before AutoML and never read by the loop. Headline TI is reported on TEST. Floor: require ≥ **k=15 test shots** and ≥ **N=3000 observed CES_TI** test samples; if unmet, headline downgrades to the **pre-registered fallback** (selection-val with an explicit optimism caveat).
- **PR4 — Bootstrap composition:** ONE canonical TEST split; **shot-clustered paired bootstrap** on per-sample `SE_model − SE_best_interp` (resample whole shots, 10k resamples), 95% CI excluding 0 = PASS. The ≥5 split seeds are used only to show the **win sign is stable across splits** (secondary), not pooled into the headline CI.

---

## Acceptance Criteria (testable)
1. `ces_prediction/baselines_interpolation.py` implements pure fns `interp_linear`, `interp_pchip` (`scipy.interpolate.PchipInterpolator`), `ar_local` (past-only causal ref; **AR order pinned = 1 / local-slope, documented**), optional `gp_matern` (**Matérn-3/2, bounded support; skipped+logged if scipy/sklearn unavailable**), each over `(neighbor_times, neighbor_values, target_time)`.
2. `build_neighbor_set(file_array, time_col, target_col, row_index)` excludes `row_index`, drops per-target NaN, and **refuses (NaN) across ≥0.5 s gaps** (mirrors `dataset.py:271`); reads only from `file_array` (Principle 3); asserts target column still contains NaN (guards `dataset.py:91-95`).
3. `build_clean_val_subset()` factored into `evaluate.py` (off the scoring path) and imported by both `evaluate.py` and `compare_baselines.py` — single source of truth (Principle 4).
4. `compare_baselines.py` reproduces the exact clean-eval Subset via the shared helper, runs model + all baselines in one pass from `file_arrays[file_idx]` at `row_index`, denormalizes to physical units (`evaluate.py:185-187`), applies the **PR2** population/fallback, and writes `comparison_metrics.json` (NOT `eval_metrics.json`).
5. Per target, physical units: `rmse_model`, `rmse_<each baseline>`, `skill_vs_interpolation = 1 − MSE_model/MSE_PCHIP` (PR1).
6. **SUCCESS (CES_TI):** on the TEST split (PR3), shot-clustered paired bootstrap CI on `SE_model − SE_PCHIP` excludes 0 in the model's favor (PR4); win sign stable across ≥5 split seeds. Emitted as a programmatic PASS/FAIL.
7. **CES_VT:** reported with identical machinery; expected non-win framed as the T_i↔V_rot asymmetry; **VT reported as descriptive (val) if the TEST VT floor is unmet** (documented, per the power tradeoff).
8. Gap-length stratification reuses `analyze_gap.py` `BIN_EDGES_MS` + `_dt_since_last_obs`; per-Δt-bin skill-vs-PCHIP per target; bins below min-n suppressed; per-bin future-vs-fallback fraction reported (PR2).
9. Multi-seed protocol: **PRIMARY = vary split seed** (≥5; CI reflects shot-to-shot generalization). `CES_INIT_SEED` added to `train.py` (default `=CES_SEED`) so model-init can vary on a fixed split — **SECONDARY stability check only**.
10. MNAR disclosure printed/written **next to the headline** number (observed-points-only optimistic bound), plus the note that interpolation uses full-shot neighbors while the model uses a ≤`window_size` history.
11. Final fixed model = AutoML `best_model.py` snapshot with pinned `WINDOW_SIZE`/config; loop documented as the search method (appendix).
12. `python -m pytest -q` green; new baseline unit tests pass; `test_architecture.py` unchanged; **before/after refactor, `evaluate.py` produces byte-identical `eval_metrics.json`** on a fixed seed/model (regression guard).

---

## Implementation Steps
0. **Literature/domain check (fold into write-up):** standard fusion-diagnostic gap-filling = linear, monotone **PCHIP** (avoids spline ringing at ELM/sawtooth), **GP** (Chilenski 2015; Ho 2019), IAR(1) for irregular series (Eyheramendy 2018); skill score `1−MSE_model/MSE_baseline` (Murphy 1988). Record citations (via document-specialist) in `RESEARCH_PLAN.md` + thesis methods.
1. **Refactor (additive, off-path):** extract `build_clean_val_subset()` in `evaluate.py`; verify byte-identical `eval_metrics.json` (AC12).
2. **`baselines_interpolation.py`** (AC1-2) + unit tests.
3. **3-way split:** extend `split_indices_by_file` + fixed-split persistence (`train.py:62-101,161-201`) to reserve a TEST split untouched by AutoML; note normalization re-fits on the new train files; document split regeneration (delete stale `fixed_*_split.csv`).
4. **`CES_INIT_SEED`** in `train.py:274` (default `=CES_SEED`), contract-safe.
5. **`compare_baselines.py`** (AC3-5, PR1-PR2) → `comparison_metrics.json`; emit per-shot per-arm squared errors for the bootstrap.
6. **Gap-stratified comparison** (AC8) — extend `analyze_gap.py` or a sibling.
7. **`run_multiseed.py`** (AC9, PR3-PR4): PRIMARY vary-split into throwaway dirs; canonical TEST-split shot-clustered paired bootstrap; secondary init-seed stability.
8. **Final fixed-model run** (AC11) + `HANDOFF.md`/`RESEARCH_PLAN.md` write-up with MNAR + neighbor-access disclosures.

---

## Risks and Mitigations
| # | Risk | Mitigation |
|---|------|-----------|
| 1 | Index-space misalignment | Read only `file_arrays[file_idx]`; automated identical-index assertion |
| 2 | Leakage (baseline sees own target) | `build_neighbor_set` excludes row_index + per-target NaN; unit test |
| 3 | Future-neighbor availability biases population | **PR2** fallback-to-persistence so the arm is defined everywhere the model is; report fraction per bin |
| 4 | Selection-on-test (AutoML selects on val) | **PR3** held-out TEST split; pre-registered fallback if floor unmet |
| 5 | Init-only CI misrepresents generalization | **PR4/AC9** PRIMARY vary-split; init demoted to stability check |
| 6 | Winner's curse picking best baseline | **PR1** pre-register PCHIP / Bonferroni |
| 7 | Harness drift from evaluate.py | **AC3** shared helper + **AC12** byte-identical regression guard |
| 8 | Cross-gap interpolation | ≥0.5 s refusal mirroring `dataset.py:271` |
| 9 | VT underpowered (esp. with TEST carve-out) | **AC7** VT descriptive-on-val fallback; quantify against data before committing a VT CI |
| 10 | GP cost / missing dep | bounded support; optional; skip+log if unavailable |
| 11 | MNAR optimism | **AC10** disclosed next to headline, both directions |

## Verification Steps
1. Unit tests: linear/PCHIP recover known functions; PCHIP does not overshoot a synthetic step (justifies the choice).
2. Leakage test: sentinel at target row → every baseline independent of it.
3. Identical-sample test: scored `(file,row_index)` set byte-identical across arms.
4. Gap-refusal test: neighbors across 0.6 s → NaN → fallback (PR2).
5. Persistence regression: harness persistence RMSE matches `evaluate.py` `rmse_persistence` on the same split.
6. Refactor regression: byte-identical `eval_metrics.json` before/after (AC12).
7. Index cross-check (Option B as audit): ~20 samples, confirm `file_arrays[file_idx][row_index, time_col]` matches independently re-derived time.
8. Contract intact: `pytest -q` green; `test_architecture.py` unchanged.

---

## ADR
- **Decision:** Add an additive, off-scoring-path comparison harness (Option A + shared helper) that pits the fixed AutoML-selected model against pre-registered conventional interpolation baselines (PCHIP headline) on a held-out TEST split, with a shot-clustered paired bootstrap CI, gap-stratified, multi-seed by split.
- **Drivers:** sample-alignment correctness; a CI that supports generalization; a defensible non-strawman baseline.
- **Alternatives considered:** B (CSV re-parse) — invalidated (filtered-array index space); C (edit evaluate.py) — invalidated (AutoML scoring path), its drift-immunity recovered via the shared helper; init-only multi-seed — rejected (misrepresents generalization); val-only headline — rejected for the headline (selection bias), allowed only as pre-registered fallback.
- **Why chosen:** only Option A guarantees identical index space with zero risk to the search loop; the shared helper removes its sole weakness (drift).
- **Consequences:** a 3-way split shrinks train + VT power (accepted; VT may be descriptive-on-val); extra compute for vary-split seeds; new harness + tests to maintain.
- **Follow-ups (open questions):** exact VT power against real data; whether to also report model-vs-best-of-ladder (Bonferroni) alongside the PCHIP headline; neighbor support window per method (fixed a priori); min-n per Δt bin.

## Changelog (consensus improvements applied)
- Architect: added held-out TEST split (Req); PRIMARY=vary-split (Req); shared `build_clean_val_subset()` (Req); MNAR/neighbor-access disclosure.
- Critic: pre-registration block PR1-PR4 (best-interp rule; population fallback; test floor+fallback; bootstrap composition); byte-identical eval_metrics regression guard; pinned AR/GP hyperparameters + GP availability fallback.
