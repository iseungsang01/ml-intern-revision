# Work Plan (DRAFT v3 — RALPLAN-DR short mode): High-Variability (Peak) CES Reconstruction Evaluation, Examples & Cause Analysis

- Plan ID: ces-peak-recon-plan
- Source spec: `.omc/specs/deep-interview-ces-peak-recon-eval.md` (PASSED, ambiguity 19%)
- Date: 2026-06-23
- Status: DRAFT v3 (revised after Architect re-review of v2: SOUND-WITH-CHANGES) for consensus re-review
- Type: brownfield, eval/visualization/analysis ONLY (no training/model/contract changes)

---

## Requirements Summary

The thesis already proves the model beats conventional interpolation **globally** (PCHIP headline, shot-clustered bootstrap CI lower bound > 0) via `compare_baselines.py` + `bootstrap_compare.py`. This work goes one step further **without retraining or changing the model**:

> The same model trained with ordinary **uniform masked-MSE (NO peak weighting)** reconstructs held-out real CES values better than interpolation **especially in high-local-activity (high-variability) neighborhoods** (ELM/sawtooth are illustrative examples, NOT a detection criterion). The headline metric flags such neighborhoods from a **target-independent proxy** (CES-neighbor activity, excluding the target row) — a conservative regional proxy, NOT pointwise extrema. Prove it (1) quantitatively with a peak-subset skill metric + shot-clustered bootstrap CI, (2) with a fair best/median/worst example spectrum (time-series figures), (3) with a single ablation (zero fast diagnostics → high-variability skill drops).

Peak-weighted training is an explicit **NON-GOAL** (deferred to the AutoML loop's auto-recommendation, AC8). This is layered work that must not regress the existing significance machinery.

**Resolved design decisions (no longer open):**
- **Module home:** Option A standalone `ces_prediction/peak_analysis.py` **plus** a minimal Option-B additive hook in `compare_baselines.py` (exactly ONE new npz key per target, `{name}_is_peak`). Committed.
- **Peak thresholds:** pin conservative defaults now AND run a 2–3 setting sensitivity sweep recorded in `HANDOFF.md`.
- **Sufficiency gate:** specify BOTH a min peak-ROW count and a min peak-SHOT count; the SHOT count is the binding gate for CI validity.
- **matplotlib:** optional `[viz]` extra in `pyproject.toml` + lazy guarded import (`Agg` backend); figures skippable with `--no-figures`. Never a core dependency.

### Two peak families (critical clarification — resolves the iteration-1 `plan:85` circularity)

There are **two distinct peak definitions**, and the headline rests on the input-only one:

- **Family (i) INPUT-DEFINED (high-local-activity) neighborhoods = PRIMARY / defensible headline.** A timestep is flagged using ONLY the neighbor/input set that **excludes `row_index`** (the target's own realized value is never read). Because the flag does not depend on `CES[idx]`, a model win here is NOT mechanically guaranteed and is a fair test of the thesis claim.
  - **Precise wording of the headline claim (fix 3):** these signals select **high-local-activity (high-variability) NEIGHBORHOODS** — a conservative, target-independent *proxy* for high-variability regions — **NOT pointwise extrema**. (A smooth-but-fast ramp has a large neighbor-bracket slope at every interior point yet contains no peak; the proxy will flag it. That is acceptable and honest: the claim is "the model reconstructs observed CES better than interpolation **in high-local-activity neighborhoods**," not "the model nails pointwise extrema.")
  - **Two DISTINCT sub-signals, reported SEPARATELY (do not merge):**
    - **(i-a) CES-neighbor activity** — large neighbor-bracket slope across `row_index` and/or high local CES-neighbor variance (from `build_neighbor_set`, excludes `row_index`). Measures **high CES variability** in the neighborhood. This is the sub-signal for the thesis claim.
    - **(i-b) Fast-diagnostic activity** — high local BES/ECEI input variance in the window. Measures **high diagnostic activity**, a DIFFERENT claim than high CES variability. Report it as its own labeled sub-family; never fold it into the "CES variability" headline number.
- **Family (ii) OBSERVED-TARGET-DEFINED peaks = SECONDARY.** A timestep is flagged using the realized observed `CES[idx]` (local extremum / large `|ΔCES|` at `idx`). Reported as **"reconstruction quality on observed high-variability points"** with an explicit **selection-bias caveat written into the JSON and the report**: selecting peaks by deviation-from-neighbors mechanically penalizes neighbor-smooth interpolation, so this number is optimistic and is NOT the thesis headline.

Explicit statement of which definition reads `CES[idx]`:
- `detect_peak_rows_input_only(...)` → **does NOT read `CES[idx]`** (uses `build_neighbor_set` semantics, excludes `row_index`). → Family (i), headline. Returns sub-signals (i-a) CES-neighbor activity and (i-b) fast-diagnostic activity separately.
- `detect_peak_rows_target(...)` → **DOES read `CES[idx]`**. → Family (ii), secondary, caveated.

Key grounding facts (existing code, verified by reading):
- Clean (non-augmented) val subset is the single source of truth: `evaluate.build_clean_val_subset` (`ces_prediction/evaluate.py:95-123`), reused by `compare_baselines.py:64-66`. File-level split, no shot leakage.
- Per-target masked keep + persistence available: `_persistence_from_history` (`evaluate.py:55-77`); keep mask `(target_mask>0.5) & has_obs` (`compare_baselines.py:112`, `evaluate.py:214`).
- **`valid` is NARROWER than `keep`:** in `compare_baselines.py:168-171`, `valid = k.copy(); for m in methods: valid &= ~np.isnan(base_phys[m][:, t])`. The per-sample SE/shot/dt boot arrays are sliced by `valid` (`compare_baselines.py:218-222`). The peak SE array MUST be sliced by the same `valid` mask.
- No-leakage interpolation neighbor set (excludes target's own row, contiguous block `<0.5s`): `baselines_interpolation.build_neighbor_set` (`baselines_interpolation.py:61-75`), block bounds (`baselines_interpolation.py:46-58`), `GAP_SECONDS=0.5` (`baselines_interpolation.py:39`). PCHIP/linear predictors with persistence fallback (`baselines_interpolation.py:87-108`).
- Per-sample squared-error npz archive (bootstrap input) written by `compare_baselines.py:217-229`: keys `{name}_shot`, `{name}_dt_ms`, `{name}_se_model`, `{name}_se_pchip`, `{name}_se_linear` — all sliced by `valid`.
- Shot-clustered paired bootstrap, 95% CI, gate `skill_lo > 0`, seed 12345, B=10000: `bootstrap_compare._bootstrap` (`bootstrap_compare.py:35-64`), `_per_shot_sums` (`bootstrap_compare.py:26-32`). It resamples **whole shots** (`bootstrap_compare.py:42,58`), so **cluster (shot) count governs CI validity**. The existing `> 50` guard (`bootstrap_compare.py:96`) is on **ROWS** (`small = data[f"{name}_dt_ms"] <= 15.0`), NOT shots — do NOT copy it as a shot guard.
- Paired-diff CI already produced by `_bootstrap`: `paired_diff_point`, `paired_diff_ci95` (`bootstrap_compare.py:40,56,62`).
- Ablation hooks live in **evaluate.py, not compare_baselines.py**: `VALID_ABLATIONS=("none","no_history","no_fast")` (`evaluate.py:80`), `apply_ablation` zeroes BES/ECEI/MC for `no_fast` (`evaluate.py:83-92`). `compare_baselines.py` has NO ablation path.
- Target timestep fully masked: `dataset.py:381-382`. Per-target independent missingness; contiguous block via `time` delta `<0.5`.
- **Researcher read path (critical for AC8 — corrected in v3):** the loop's `build_briefing` (`automl_agent_loop.py:413-443`) reads the **IN-MEMORY** `result.get("eval")` captured at `automl_agent_loop.py:326` (`"eval": eval_report`), where `eval_report` is the in-memory return of `run_clean_eval` (`automl_agent_loop.py:311`); it NEVER re-reads `eval_metrics.json` from disk (`:414`). Critically, **`run_evaluation` only calls `run_clean_eval` + `run_comparison`** (`automl_agent_loop.py:309-313`) and assembles `result` at `:324-332` — it **never invokes `peak_analysis.py`**. Therefore a peak block merged onto the on-disk `eval_metrics.json` would NOT reach the briefing. `comparison_metrics.json` reaches ONLY `comparison_skill` scoring (`automl_agent_loop.py:117-133,290,318`), NOT the researcher prompt. The `program` text is injected verbatim into the prompt at `automl_agent_loop.py:475`. **So AC8 requires an additive in-loop edit (Option 1): compute the peak block inside `run_evaluation` and merge it into `eval_report` IN MEMORY before `result` is assembled, plus surface it in `build_briefing`, plus the standing `program.md` rule.**
- Output dir convention: `CES_OUTPUT_DIR` (default `ces_prediction/`); all JSON/npz artifacts land there (`compare_baselines.py:51,228-233`, `evaluate.py:129,258`).
- No plotting code exists anywhere; `matplotlib` is NOT in `pyproject.toml` dependencies (`pyproject.toml:7-15`) — AC6 must add it as a `[viz]` extra.

---

## Acceptance Criteria (restated, each testable; AC2/AC3/AC4/AC7/AC8 reworded per review)

- **AC1 — baseline preserved (NUMERIC non-regression gate).** Global `skill_vs_pchip` and the shot-clustered 95% CI gate (`skill_lo>0`, `bootstrap_compare.py:63`) unchanged after the `compare_baselines.py` hook.
  - *Testable:* (a) `python -m pytest -q tests/test_baselines_interpolation.py tests/test_bootstrap_compare.py` green; (b) **numeric**: snapshot `comparison_metrics.json` (`per_target.*.skill_vs_pchip`, `rmse_model`, `n`) and `bootstrap_summary.json` (CI bounds, `pass`) on real data BEFORE the change, then assert float-equality of those exact keys AFTER (Verification step 1).
- **AC2 — peak detection (two families; input-only is headline).** Deterministic, no-leakage detection. `detect_peak_rows_input_only` (Family i, excludes `row_index`, headline) and `detect_peak_rows_target` (Family ii, reads `CES[idx]`, secondary/caveated). Documented default params; within contiguous blocks (`<0.5s`).
  - *Testable:* unit tests on synthetic series: flat → no peaks; injected neighbor-bracket slope → Family-(i) detected; injected spike at idx → Family-(ii) detected; **labels invariant to row reordering within a block**; **robust to NaN interior rows**; **no leakage across `>=0.5s` gaps**; fixed params → identical mask.
- **AC3 — peak-subset metrics (per family/sub-signal, per target).** Per-target `peak_skill_vs_pchip` (and vs linear), `peak_rmse_model`, `peak_rmse_pchip` in **physical units**, for the headline sub-signal (i-a CES-activity), the separate sub-signal (i-b fast-activity), and Family (ii), emitted to JSON; Family (ii) carries the selection-bias caveat string; (i-a) carries the "high-local-activity neighborhood, not pointwise extrema" definition note.
  - *Testable:* JSON `peak.input_only.ces_activity` (headline), `peak.input_only.fast_activity` (separate), and `peak.observed_target` blocks per target with those keys; finite when `n_peak_rows>0`; the headline block computed on the `valid`-masked npz peak samples only; Family (ii) block contains `selection_bias_caveat`; (i-a) contains `definition` note.
- **AC4 — peak significance (shot-bound dual-sufficiency guard).** Shot-clustered bootstrap 95% CI on the peak subset per target/family; verdict `peak CI lower bound > 0`. Binding gate is **distinct peak SHOT count** (`N_MIN_PEAK_SHOTS`), with a secondary min peak-ROW count.
  - *Testable:* JSON has `peak_skill_ci95` + `pass` per family/target reusing `bootstrap_compare._bootstrap`; when `n_shots < N_MIN_PEAK_SHOTS` (or `n_rows < N_MIN_PEAK_ROWS`), emit `insufficient_shots: true` / `insufficient_rows: true` and `pass` forced false, no crash. CES_TI/CES_VT asymmetry reported as-is.
- **AC5 — example selection (exact, deterministic).** Best/median/worst by per-sample skill (improvement over interpolation), k per band (default k=3, configurable 3–5), exact index math + stable tie-break.
  - *Testable:* unit test: for fixed per-sample skills, selection returns the exact expected indices; deterministic; median band uses the specified index math (Step 5).
- **AC6 — visualization.** Per selected example, a CES time-series figure overlaying observed truth / model prediction / PCHIP (and linear) with the peak point highlighted, saved via matplotlib `savefig`. Figures are generated for the headline set (i-a) at minimum; (i-b)/(ii) figures optional.
  - *Testable:* the figure step produces N = peak_sets×bands×targets×k `.png` files in `CES_OUTPUT_DIR/peak_figures/` (peak_sets ≥ 1, the headline i-a); `Agg` backend; no `plt.show`; `--no-figures` skips it; import guarded so non-viz paths never require matplotlib.
- **AC7 — cause-analysis ablation (own forward pass, paired-shot delta, OOD-disclosed).** `no_fast` (zero BES/ECEI/MC) peak-subset skill reported alongside `full`; a significant drop demonstrates multimodal usage. Ablated arm reuses the IDENTICAL keep+peak subset; paired-shot delta with pinned sign convention.
  - *Testable:* JSON has `peak.input_only.full` vs `peak.input_only.no_fast` per target with each arm's skill + the paired-diff CI. Significance of "removing fast diagnostics hurts (full is better)" is judged by **`paired_diff_ci95[0] > 0`** (the diff LOWER bound > 0), where `paired_diff = SE_no_fast − SE_full` (positive ⇒ full better). **`_bootstrap`'s own `pass`/`skill` fields MUST be disregarded for the ablation arm** (with this argument order they test the opposite direction — see Step 4). RISKS discloses the OOD-zeros confound.
- **AC8 — loop integration (reaches the researcher IN-LOOP, Option 1: real numbers).** An additive edit in `run_evaluation` (`automl_agent_loop.py`) computes the peak block and merges it into `eval_report["per_target"][name]["peak"]` IN MEMORY before `result` is assembled (`:324-332`); `build_briefing` surfaces per-target peak skill/CI from `result["eval"][...]["peak"]` into the briefing STRING; `program.md` carries the standing steering rule. Default scoring (`comparison_skill`) and keep/discard gate numerically unchanged — peak metrics inform the RESEARCHER (model.py rewrite) only, NEVER the gate.
  - *Testable (REAL path, not synthetic-only):* (i) a test that drives `run_evaluation` (with stubbed subprocess/eval) populates `result["eval"]["per_target"][name]["peak"]`; (ii) `build_briefing(result)` returns a string CONTAINING the per-target peak skill/CI fields; (iii) `program.md` contains the peak-steering paragraph; (iv) unit test asserts `comparison_skill` returns an identical value with vs without the `peak` keys present (gate unaffected). A purely-synthetic `result["eval"]` test MAY supplement but MUST NOT be the only AC8 evidence.
- **AC9 — tests & docs.** New code unit-tested; `python -m pytest -q` green; if behavior changed, one smoke run recorded; sensitivity sweep + peak results in `HANDOFF.md`; major findings → `PROJECT_KNOWLEDGE.md`.
  - *Testable:* full suite green; HANDOFF/PROJECT_KNOWLEDGE diffs present; smoke + sweep recorded.

---

## Guardrails

### Must Have
- New module `ces_prediction/peak_analysis.py` (detection both families, peak-subset metric assembly + bootstrap CI, example selection, figures, AC7 ablation forward pass) — **reusing** `baselines_interpolation`, `bootstrap_compare._bootstrap`, `evaluate.build_clean_val_subset`, `evaluate._persistence_from_history`, `evaluate.apply_ablation`.
- **Minimal hook in `compare_baselines.py`:** add exactly ONE new npz key per target, `boot[f"{name}_is_peak"] = is_peak[valid]` (input-only family, the headline), aligned to the per-target `valid` mask (`compare_baselines.py:168-171`), placed alongside the existing boot writes (`compare_baselines.py:217-222`). This makes the **non-ablated peak SEs the byte-identical headline SEs**. The standalone module computes figures/diagnostics/the secondary family/the AC7 ablated arm but is NOT the source of the headline significance numbers.
- **Additive in-loop edit to `automl_agent_loop.py:run_evaluation`** (NOT the data/model contract, NOT training logic, NOT AC1-critical): after `run_comparison` (`:313`), call a thin `run_peak_eval(env)` wrapper (runs/imports `peak_analysis`) and merge the peak block into `eval_report["per_target"][name]["peak"]` IN MEMORY before `result` is assembled (`:324-332`). Plus ~2 lines in `build_briefing` and the `program.md` paragraph.
- All evaluation on the **clean non-augmented val split**, **physical units**, **per target**, **observed-CES-only** (MNAR honored), **no leakage**.
- Deterministic: seed 12345 for bootstrap; pinned numeric peak thresholds; stable example selection.
- Additive only: peak block merged into the in-memory `eval_report` (briefing path) + `program.md` standing rule; `comparison_skill` and the keep/discard gate numerically unchanged.

### Must NOT Have
- NO edits to `model.py` (forward/architecture), `train.py` loss, or the data/model contract.
- NO peak-weighted training / objective change (deferred non-goal).
- NO MC/ECEI/Dα ELM detector — peak = observed/input CES-signal high-variability only.
- NO new/synthetic data ([[no-fake-data]]); real CSVs only, fail loudly if `CES_DATA_DIR` missing.
- NO git commits/pushes; NO cherry-picking (best/median/worst spectrum mandatory).
- NO change to `comparison_skill`/default loop scoring or the keep/discard gate; the `run_evaluation` edit feeds the RESEARCHER only, never the gate; NO copying the `>50` ROW guard as a shot guard.
- matplotlib NEVER a core dependency.

---

## Implementation Steps

### Step 1 — Peak detection, two families + two input sub-signals (AC2; fixes 1, 3)
- New file `ces_prediction/peak_analysis.py`.
- `detect_peak_rows_input_only(file_array, time_col, target_col, *, gap=0.5, slope_z=..., neigh_var_pct=..., bes_ecei_var_pct=...) -> {"ces_activity": bool mask, "fast_activity": bool mask}` — **headline, Family (i)**. Within each contiguous block (reuse `baselines_interpolation.contiguous_block_bounds`, `:46-58`), classify `row_index` using ONLY the excluded-`row_index` neighbor/input set (`build_neighbor_set`, `:61-75`). Never reads `CES[idx]`. Returns TWO DISTINCT sub-signals SEPARATELY (fix 3 — do not merge):
  - **(i-a) `ces_activity`** — large neighbor-bracket slope across `row_index` (top `slope_z`) and/or high local CES-neighbor variance (top `neigh_var_pct`). This is the **headline sub-signal** (high CES variability). NOTE this selects high-local-activity NEIGHBORHOODS, a conservative regional proxy, NOT pointwise extrema (documented in the JSON `definition` field).
  - **(i-b) `fast_activity`** — high local BES/ECEI input variance in the window (top `bes_ecei_var_pct`). Measures high DIAGNOSTIC activity (a different claim); reported as its own labeled sub-family, never folded into the CES-variability headline.
- `detect_peak_rows_target(file_array, time_col, target_col, *, gap=0.5, diff_z=..., roll_window=..., top_pct=...) -> bool mask` — **secondary, Family (ii)**. Flags local extremum / large `|ΔCES|` / high rolling-std **at `idx` using `CES[idx]`**. Carries a documented `selection_bias_caveat`.
- The headline `is_peak` written to the npz (Step 2) is sub-signal **(i-a) `ces_activity`**. Sub-signal (i-b) and Family (ii) are computed in-module for separate reporting.
- `PEAK_PARAMS` dict pins conservative defaults (e.g. i-a: `slope_z=2.5`, `neigh_var_pct=0.10`; i-b: `bes_ecei_var_pct=0.10`; target: `diff_z=2.5`, `roll_window=5`, `top_pct=0.10`). Pure numpy, no RNG. All logged into JSON. (defaults pinned now; sweep in Step 8.)

### Step 2 — Byte-aligned npz hook for the headline subset (AC1, AC3, AC4; fix 5)
- In `compare_baselines.py`, inside the per-target loop after `valid` is built (`:168-171`), compute `is_peak` = sub-signal **(i-a) `ces_activity`** (Family i, input-only, headline) per kept sample and write `boot[f"{name}_is_peak"] = is_peak[valid]` next to the existing boot writes (`:217-222`). **Slice by `valid`, not `keep`** — `valid` is narrower (drops samples where any arm is NaN) and the SE arrays use `valid`; misalignment would desync the bootstrap. Assert `len(is_peak[valid]) == len(boot[f"{name}_se_model"])` in code.
- This is the ONLY load-bearing edit to `compare_baselines.py` (an additive `report["per_target"][name]["peak"]` summary is allowed but the npz key is the binding change). Sub-signal (i-b) and Family (ii) are NOT written to the npz — they are recomputed in `peak_analysis.py` for separate reporting only (they are not the headline significance numbers).

### Step 3 — Peak-subset metric + bootstrap CI, dual-sufficiency guard (AC3, AC4; fix 4)
- In `peak_analysis.py`, `run_peak_eval(...)`: load `comparison_errors_{split}.npz`, mask each target's SE/shot arrays by `{name}_is_peak`, compute per-target `peak_rmse_model`, `peak_rmse_pchip`, `peak_skill_vs_pchip = 1 - MSE_model/MSE_pchip` (and vs linear) in physical units (mirrors `compare_baselines.py:174-191`).
- Bootstrap: `bootstrap_compare._bootstrap(shot_peak, se_model_peak, se_pchip_peak, np.random.default_rng(12345))` → `peak_skill_ci95`, `pass`.
- **Dual sufficiency guard (shot-bound):** count distinct peak shots `n_shots` and peak rows `n_rows`. `N_MIN_PEAK_SHOTS = 15` (binding gate) and `N_MIN_PEAK_ROWS = 200` (secondary). If `n_shots < N_MIN_PEAK_SHOTS` → `insufficient_shots: true`, `pass` forced false. If `n_rows < N_MIN_PEAK_ROWS` → `insufficient_rows: true`. Never crash.
  - *Rationale for N_MIN_PEAK_SHOTS = 15:* `_bootstrap` resamples whole shots (`bootstrap_compare.py:42`), so cluster count, not row count, governs percentile-CI stability; with <~15 clusters the CI is over-narrow/unstable. 15 is a conservative shot floor; raise if the peak subset spans many shots. Explicitly NOT the `>50` ROW guard at `bootstrap_compare.py:96`.
- Repeat for Family (ii) via `detect_peak_rows_target` (computed in-module; labeled secondary with the caveat).

### Step 4 — AC7 ablation: own forward pass, sample-for-sample paired delta (AC7; fixes 2, 4, 6)
- `compare_baselines.py` has NO ablation path, so the `no_fast` arm gets its **own forward pass in `peak_analysis.py`**: rebuild the identical eval subset via `evaluate.build_clean_val_subset` and apply `evaluate.apply_ablation(ablate="no_fast", ...)` (`evaluate.py:83-92`). Persistence/interp arms come from real data (ablation only zeroes BES/ECEI/MC, mirroring `evaluate.py:181-184`).
- **Pairing guard (fix 4) — do NOT trust independently recomputed masks to coincide with the headline npz.** Load `comparison_errors_{split}.npz`'s `{name}_shot`, `{name}_is_peak` (and `{name}_se_model` = the FULL-arm headline SEs). Align the ablation forward pass to **those EXACT indices**: iterate the eval subset in the same `compare_baselines.py:99-141` order, and for each target either (preferred) index into the loaded `{name}_shot`/`{name}_is_peak` arrays directly, OR recompute `valid`/`is_peak` and **assert element-wise equality** (`np.array_equal`) against the npz arrays before computing any delta. If they differ, fail loudly (no silent desync). `se_full` for the delta IS the npz `{name}_se_model` (byte-identical to the headline), not a re-forward.
- The ablated `se_no_fast` is the per-sample SE of the `no_fast` forward over the SAME peak subset (same `shot`/`is_peak` slice).
- **Sign convention pinned (fix 2):** call `_bootstrap(shot_peak, se_model=se_no_fast, se_base=se_full, np.random.default_rng(12345))`. Then `_bootstrap` internally computes `diff = SE_no_fast − SE_full` (`bootstrap_compare.py:50`) and `skill = 1 − MSE_no_fast/MSE_full` with `pass = skill_lo>0` (`:63`) — **with this argument order those `skill`/`pass` fields assert the OPPOSITE of what we want and MUST be ignored.** Judge significance ONLY by **`paired_diff_ci95[0] > 0`** (the diff LOWER bound strictly above 0 ⇒ `SE_no_fast > SE_full` significantly ⇒ removing fast diagnostics significantly hurts at peaks ⇒ the model genuinely uses the multimodal signal). State this explicitly so an implementer cannot report ablation significance backwards.
- Report `peak.input_only.full` skill, `peak.input_only.no_fast` skill, and `paired_diff_point` + `paired_diff_ci95` per target; honest reporting. (OOD confound disclosed in RISKS.)

### Step 5 — Example selection (AC5; fix 7)
- `select_examples(per_sample_skill, shot_ids, row_indices, k=3) -> dict[band -> list[idx]]`. Per-sample skill = `SE_pchip - SE_model` (positive ⇒ model better) on the peak subset.
- Sort ascending by the composite key `(skill, shot_id, row_index)` (stable tie-break). Let `n = len(skill)`, `order` = argsort with that key.
  - `worst  = order[0:k]`
  - `best   = order[n-k:n]`
  - `median = order[m - (k//2) : m - (k//2) + k]` where `m = n // 2`
- All three bands always emitted (failures included in `worst`). Selected descriptors (shot file, row_index, target, time, skill, family) persisted to `peak_metrics.json["examples"]`.

### Step 6 — Figures (AC6; fix d)
- Add `matplotlib>=3.7` to a NEW `[viz]` optional-dependencies group in `pyproject.toml` (`pyproject.toml:17-20`). Never core.
- `plot_example(...)` in `peak_analysis.py`: lazy `import matplotlib; matplotlib.use("Agg")` inside the figure function (guarded with a clear error if `[viz]` missing). Overlay observed truth (scatter), model prediction at the eval timestep, PCHIP + linear curves over the example's contiguous block (reuse `build_neighbor_set`/`contiguous_block_bounds`); highlight the peak timestep. `savefig` → `CES_OUTPUT_DIR/peak_figures/{family}_{target}_{band}_{rank}.png`. No `plt.show`. `--no-figures` flag (and/or env) skips entirely so loop runs stay headless and cheap.

### Step 7 — Loop / researcher exposure, IN-LOOP Option 1 (AC8; BLOCKING fix 1)
> v2's "write back to `eval_metrics.json` on disk" does NOT work: the loop never invokes `peak_analysis.py` (`run_evaluation` only runs `run_clean_eval`+`run_comparison`, `automl_agent_loop.py:309-313`), and `build_briefing` reads the IN-MEMORY `result["eval"]` captured at `:326`, never re-reading the file (`:414`). v3 uses the real in-memory path.
- (a) **`run_peak_eval(env)` wrapper** in `peak_analysis.py` (importable): returns the per-target peak block dict (Family i headline incl. sub-signals i-a/i-b, Family ii secondary, AC7 full-vs-no_fast). It runs against the same `env`/`CES_OUTPUT_DIR`, reusing the `comparison_errors_{split}.npz` that `run_comparison` just produced.
- (b) **Additive edit to `automl_agent_loop.py:run_evaluation`:** after `comparison_report = self.run_comparison(env)` (`:313`), call `peak_block = run_peak_eval(env)` and merge IN MEMORY: for each target, `eval_report["per_target"][name]["peak"] = peak_block[name]` — BEFORE the `result` dict is assembled at `:324-332`. Wrap in try/except so a peak failure degrades gracefully (logs, sets `peak=None`) without failing the iteration or the gate.
- (c) **~2 lines in `build_briefing`** (inside the `for name in TARGET_NAMES` loop, `automl_agent_loop.py:422-429`): when `stats.get("peak")` is present, append a line with per-target peak `skill_vs_pchip` + `peak_skill_ci95` (and the AC7 ablation delta if present) to the briefing STRING.
- (d) **Peak-aware steering paragraph in `program.md`** (injected verbatim at `automl_agent_loop.py:475`): standing guidance — "A peak metric reports skill on high-local-activity neighborhoods; if input-defined peak skill is weak / its CI straddles 0 (esp. CES_VT), consider proposing a peak-weighted loss." This is the deferred-recommendation hook (AC8 intent): live numbers via the briefing + a standing rule via `program.md`.
- Keep `comparison_skill` and the keep/discard gate numerically unchanged (the gate still reads only `skill_vs_pchip` via `comparison_skill`, `automl_agent_loop.py:117-133,318`); peak metrics inform the researcher's model.py rewrite ONLY.

### Step 8 — Tests, sweep & docs (AC1, AC9; fix 3 + determinism test)
- New `tests/test_peak_analysis.py` (CPU-only, synthetic numpy; mirror `test_baselines_interpolation.py` style):
  - peak detection both families on flat/spiky series; **determinism + invariance to row reordering within a block; robustness to NaN interior rows; no-leakage across `>=0.5s` gap**;
  - example selection exact-index test (Step 5);
  - dual-sufficiency guard (forces `pass=false` below `N_MIN_PEAK_SHOTS`);
  - **AC7 sign-convention test:** construct `se_no_fast > se_full` per sample across many shots → assert `paired_diff_ci95[0] > 0` is the significance signal, and assert the code does NOT key off `_bootstrap`'s `pass` for the ablation arm;
  - **AC8 REAL-path guards (fix 1):** a test driving `run_evaluation` with stubbed subprocess/`run_clean_eval`/`run_comparison`/`run_peak_eval` asserts `result["eval"]["per_target"][name]["peak"]` is populated AND `build_briefing(result)` returns a string containing the peak fields; plus `comparison_skill` returns an identical value with vs without `peak` keys (gate unaffected). A synthetic-`result["eval"]` test may supplement but is not the sole evidence.
- AC1 numeric snapshot/compare verification (Verification step 1).
- Run 2–3 peak-param settings and record results + chosen defaults in `HANDOFF.md`; major finding (e.g. CES_VT null at peaks) → `PROJECT_KNOWLEDGE.md`.

---

## Pre-Mortem (7 scenarios — short mode)
- **(i) Peak subset too small for a valid CI.** → Dual-sufficiency guard: `N_MIN_PEAK_SHOTS=15` binding + `N_MIN_PEAK_ROWS=200`; emit `insufficient_*`, suppress PASS, report point estimate only.
- **(ii) Circularity challenged in defense.** → Headline is Family (i) input-only (never reads `CES[idx]`); Family (ii) explicitly secondary with selection-bias caveat in JSON + report.
- **(iii) AC8 metric never reaches the researcher.** → v2's on-disk `eval_metrics.json` write-back does NOT reach the in-memory briefing (loop never calls `peak_analysis`; `build_briefing` reads `result["eval"]` at `:326`, not disk). v3 fix = additive in-loop edit to `run_evaluation` merging the peak block into `eval_report` IN MEMORY before `:324-332`, plus a `build_briefing` line, plus the `program.md` standing rule. Verified by a test that drives `run_evaluation` (stubbed) → asserts `result["eval"][...]["peak"]` populated AND `build_briefing(result)` string contains the peak fields — NOT a synthetic-only test.
- **(iv) Numeric regression in `compare()`.** → Pre/post float-equality snapshot of `comparison_metrics.json` + `bootstrap_summary.json` exact keys; only ONE additive npz key added.
- **(v) matplotlib breaks CI / install.** → `[viz]` optional extra, lazy guarded import, `Agg`, `--no-figures`; tests never import matplotlib.
- **(vi) `is_peak` misaligned to the SE arrays.** → Slice by `valid` (NOT `keep`), aligned to the existing boot writes; assert equal lengths in code + test.
- **(vii) OOD-zeros confound in AC7.** → Disclosed in RISKS; report the drop honestly; a single ablation cannot fully separate "uses fast signal" from "fed OOD zeros".

---

## Risks and Mitigations
- **Circularity / selection bias (Family ii).** Headline uses Family (i) input-only; Family (ii) caveated. (Pre-Mortem ii.)
- **Too few peak shots → unstable CI.** Dual-sufficiency guard, shot-bound. (Pre-Mortem i.)
- **CES_VT asymmetry — model may NOT beat interpolation at peaks.** Expected (model beats interp for CES_TI, not CES_VT). Report per-target honestly; a CES_VT null at peaks is a **valid reportable finding, not a bug** (spec L44).
- **OOD ablation confound (AC7).** Zeroing BES/ECEI/MC is an out-of-distribution intervention (off the training manifold); the skill drop conflates "model uses fast signal" with "model fed OOD zeros". Disclose; a single ablation cannot fully resolve it. (Follow-up: a noise/permutation ablation could address this later.)
- **MNAR optimism.** Evaluate only observed peaks; carry `mnar_caveat` (`compare_baselines.py:158`) into peak JSON.
- **Numeric regression in `compare()`.** One additive npz key; numeric snapshot gate (AC1). (Pre-Mortem iv.)
- **`is_peak` desync.** Slice by `valid`. (Pre-Mortem vi.)
- **matplotlib footprint.** `[viz]` extra, lazy/guarded, skippable. (Pre-Mortem v.)

---

## Verification Steps (with expected per-target outcomes)
1. **AC1 numeric non-regression.** With real data + a trained output dir: run `python ces_prediction/compare_baselines.py && python ces_prediction/bootstrap_compare.py`, copy `comparison_metrics.json` + `bootstrap_summary.json` to a snapshot. Apply the npz hook. Re-run. Diff with `git diff --no-index snapshot_comparison.json new_comparison.json` (and same for bootstrap) — the only differences allowed are the new additive `peak`/`is_peak` keys; `per_target.*.skill_vs_pchip`, `rmse_model`, `n`, and all bootstrap CI bounds/`pass` must be float-equal.
2. `python -m pytest -q` — full suite green incl. new `tests/test_peak_analysis.py`; existing `test_baselines_interpolation.py`, `test_bootstrap_compare.py` unchanged-green (AC1, AC2, AC5, AC8, AC9).
3. With real data: `python ces_prediction/peak_analysis.py` → `CES_OUTPUT_DIR/peak_metrics.json` (AC3/AC4/AC5/AC7) + `peak_figures/*.png` (AC6). AC8's in-memory merge is exercised separately via the `run_evaluation` test (step 5), since the standalone CLI run does not go through the loop.
4. Inspect `peak_metrics.json`: headline `peak.input_only.ces_activity` `peak_skill_vs_pchip`, `peak_skill_ci95`, `pass`/`insufficient_shots`; separate `peak.input_only.fast_activity`; Family (ii) `peak.observed_target` with caveat; AC7 `paired_diff_ci95` (judged by lower bound > 0); `examples` best/median/worst (AC5).
   - **Expected outcomes:** **CES_TI headline (i-a) peak skill > 0 expected** (consistent with the established TI result). **CES_VT MAY be non-significant** (known T_i/V_rot asymmetry) — a null there is a **reportable finding, not a failure**. AC7: for CES_TI, `paired_diff_ci95[0] > 0` expected (removing fast diagnostics hurts); report the CES_VT case honestly.
5. AC8 reach (REAL in-loop path): unit test drives `run_evaluation` (stubbed) → `result["eval"][...]["peak"]` populated; `build_briefing(result)` string contains per-target peak skill/CI; `program.md` paragraph present; `comparison_skill` and the keep/discard gate unchanged.
6. `peak_figures/*.png` count == peak_sets×bands×targets×k (headline i-a at minimum); visually confirm overlay + peak highlight (AC6).
7. Smoke + sweep: `.\ces_prediction\run_smoke_test.ps1` passes; record 2–3 peak-param settings + results in `HANDOFF.md` (AC9).

---

## RALPLAN-DR Summary

### Principles
1. **Layer, never replace** — existing CI_low>0 significance gate is the foundation; peak work adds on top, no regression.
2. **Same samples, same primitives** — reuse `build_clean_val_subset`, `build_neighbor_set`, `apply_ablation`, `_bootstrap`; headline SEs are the byte-identical compare() SEs (one npz key).
3. **Defensible, non-circular headline** — Family (i) sub-signal (i-a) input-only CES-activity (never reads `CES[idx]`; selects high-local-activity NEIGHBORHOODS, a proxy, not pointwise extrema) is the thesis claim; sub-signal (i-b) fast-activity and Family (ii) are reported separately/caveated.
4. **Honest, fair, per-target** — best/median/worst spectrum, CES_TI/CES_VT asymmetry reported truthfully, observed-CES-only (MNAR), physical units, OOD-ablation disclosed.
5. **Determinism & reproducibility** — seed 12345, pinned peak params logged + sensitivity sweep, stable example selection.

### Decision Drivers (top 3)
1. **Non-circularity of the headline** — the peak definition must not mechanically guarantee the model win (drives Family-(i) input-only as headline).
2. **AC1 numeric non-regression** — caps edits to the AC1-critical file to one additive npz key + a numeric snapshot gate.
3. **AC8 actually reaching the researcher** — peak metrics must be merged into the IN-MEMORY `eval_report` inside `run_evaluation` + surfaced in the `build_briefing` string + a `program.md` standing rule; an on-disk-only `eval_metrics.json` write would never reach the briefing.

### Viable Options
- **Option A — standalone `peak_analysis.py` only.** Pros: zero AC1 risk. Cons: headline SEs recomputed (drift risk); AC8 still needs loop wiring. — superseded by A+hook.
- **Option B — extend `compare_baselines.py` heavily.** Pros: single pass. Cons: large blast radius on the AC1-critical file. — rejected.
- **Option A + minimal Option-B npz hook + in-loop AC8 merge (CHOSEN).** Standalone module for detection/figures/sub-signal i-b/secondary family/ablation; ONE additive npz key (`{name}_is_peak`, sliced by `valid`) in `compare_baselines.py` so headline peak SEs are byte-identical; AC8 via an additive in-memory merge in `run_evaluation` + `build_briefing` line + `program.md`. Pros: byte-aligned headline, minimal AC1 surface, AC8 reaches the researcher in-loop with REAL numbers. Cons: peak metrics produced in two places (npz headline SEs + module report), and the AC8 path adds a small `automl_agent_loop.py` edit (additive, gate-neutral).
- **Option C — notebook/ad-hoc.** Rejected: not deterministic/testable/loop-consumable (violates AC8/AC9).

---

## ADR

- **Decision:** Implement peak reconstruction evaluation as a standalone `ces_prediction/peak_analysis.py` PLUS one additive `{name}_is_peak` npz key (sliced by `valid`) in `compare_baselines.py`. Headline = Family (i) input-only sub-signal (i-a) CES-activity = high-local-activity NEIGHBORHOODS (target-independent proxy, not pointwise extrema); sub-signal (i-b) fast-diagnostic activity reported separately; secondary = Family (ii) observed-target peaks (caveated). AC7 ablation runs its own `no_fast` forward pass in the module, paired sample-for-sample to the npz `{name}_se_model`/`{name}_shot`/`{name}_is_peak`, judged ONLY by `paired_diff_ci95[0] > 0` (`_bootstrap`'s `pass`/`skill` disregarded). AC8 exposure is IN-LOOP (Option 1): additive edit to `automl_agent_loop.py:run_evaluation` merges the peak block into the in-memory `eval_report` before `result` assembly (`:324-332`) + `build_briefing` line + `program.md` standing rule. matplotlib as `[viz]` optional extra, figures skippable.
- **Drivers:** non-circular defensible headline; AC1 numeric non-regression; AC8 reaching the researcher's actual read path; shot-bound CI validity.
- **Alternatives considered:** standalone-only (drift in headline SEs); heavy `compare_baselines.py` extension (AC1 blast radius); notebook/ad-hoc (non-reproducible); target-defined peaks as headline (circular).
- **Why chosen:** byte-aligned headline SEs with minimal AC1 surface; the input-only peak family rebuts the circularity objection; the in-loop in-memory merge into `eval_report` + `build_briefing` is the only path that actually reaches the researcher prompt (an on-disk `eval_metrics.json` write is never re-read, and `comparison_metrics.json` only feeds the gate).
- **Consequences:** peak metrics split across `comparison_errors_*.npz` (headline SEs) + `peak_metrics.json` (full per-family report + figures) + an in-memory `eval_report[...]["peak"]` block merged inside `run_evaluation`; additive edits to FOUR files — `compare_baselines.py` (one npz key), `automl_agent_loop.py` (`run_evaluation` merge + `build_briefing` line), `program.md` (standing rule), `pyproject.toml` (`[viz]`). The `run_evaluation` edit is additive (try/except-guarded), touches neither the data/model contract nor training logic nor the keep/discard gate. matplotlib only needed for figures.
- **Follow-ups:** if input-defined peak skill is weak / CI straddles 0 (esp. CES_VT), the `program.md` steering lets the AutoML loop auto-propose a peak-weighted loss (the deferred non-goal); revisit `N_MIN_PEAK_SHOTS` if the peak subset spans many shots; consider a manifold-respecting ablation (noise/permutation instead of zeros) to address the OOD confound.

---

## Resolved (formerly Open) Questions
- Peak thresholds → conservative defaults in `PEAK_PARAMS` now + 2–3 setting sensitivity sweep in `HANDOFF.md`.
- AC8 exposure → Option 1 (in-loop): additive merge in `run_evaluation` into the in-memory `eval_report` + `build_briefing` line + `program.md` standing rule (NOT an on-disk-only write).
- `N_MIN` → `N_MIN_PEAK_SHOTS=15` (binding, shot-clustered CI validity) + `N_MIN_PEAK_ROWS=200` (secondary).
- matplotlib → `[viz]` optional extra, lazy guarded `Agg` import, `--no-figures`.
