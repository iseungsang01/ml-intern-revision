# Deep Interview Spec: KSTAR CES nowcasting beats conventional interpolation (thesis)

## Metadata
- Interview ID: di-ces-nowcasting-thesis
- Rounds: 5 (+ Round 0 topology)
- Final Ambiguity Score: ~12%
- Type: brownfield (existing `ces_prediction/` pipeline)
- Generated: 2026-06-22
- Threshold: 0.2
- Threshold Source: default
- Initial Context Summarized: no (context drawn from this session)
- Status: PASSED

## Clarity Breakdown (brownfield weights)
| Dimension | Score | Weight | Weighted |
|-----------|-------|--------|----------|
| Goal Clarity | 0.92 | 0.35 | 0.322 |
| Constraint Clarity | 0.85 | 0.25 | 0.213 |
| Success Criteria | 0.88 | 0.25 | 0.220 |
| Context Clarity | 0.85 | 0.15 | 0.128 |
| **Total Clarity** | | | **0.882** |
| **Ambiguity** | | | **~0.118** |

## Topology
| Component | Status | Description | Coverage |
|-----------|--------|-------------|----------|
| Contribution claim | active | What the thesis argues | Goal/Acceptance defined below |
| Success metric & acceptance | active | How "better" is measured & declared | Acceptance Criteria below |
| Evidence & scope | active | Experiments/artifacts + method placement | Technical Context + Non-Goals |

## Goal
Show that a multimodal **nowcasting** model — predicting low-time-resolution KSTAR CES
`[CES_TI, CES_VT]` at a target 10 ms timestep from **simultaneous fast diagnostics
(BES/ECEI/MC) plus past CES history** — **beats conventional CES-only interpolation methods
(linear, cubic spline, AR) that are allowed to use both PAST and FUTURE CES samples**, for the
ion-temperature target `CES_TI`. Beating an interpolation baseline that even sees the future is
strong evidence that the fast diagnostics carry CES information beyond what temporal
interpolation of CES alone can recover. The model is the headline contribution; the
`T_i`↔`V_rot` asymmetry it exposes is the accompanying scientific insight.

## Constraints
- Task is **offline gap-filling**: interpolation baselines may use past+future CES; the model
  uses fast diagnostics at the target step + past CES history only. The asymmetry of information
  access is intentional and is the point of the claim.
- Real data only (641 CSVs in `data/`), GPU, train-file-only normalization, existing data/model
  contract preserved.
- Primary comparison on the **clean, non-augmented** validation split (existing `evaluate.py`
  protocol), per target, in **physical CES units**.
- Statistical rigor: multi-seed (>=5) with confidence intervals; no single-seed claims.
- The final reported model is **one fixed architecture**; the AutoML loop is the search method
  used to find it (reproducible), reported in an appendix/method section, not as the headline.

## Non-Goals
- Real-time / online deployment of the nowcaster.
- Beating interpolation on `CES_VT` (toroidal rotation). Expected NOT to beat future-using
  interpolation because fast diagnostics carry ~no rotation info at 10 ms — this under-performance
  is reported as the **asymmetry finding**, not a failure.
- Super-resolution framing; other diagnostics beyond BES/ECEI/MC; changing the data/model contract.

## Acceptance Criteria
- [ ] Conventional interpolation baselines implemented and validated: **linear interpolation,
      cubic spline, AR** over CES (past+future), plus **persistence** and **history-only**
      references. (Literature/domain check first to confirm what is standard for CES/fusion
      diagnostic gap-filling.)
- [ ] Primary metric computed per target in physical units: **RMSE_model vs RMSE_best_interpolation**
      and `skill_vs_interpolation = 1 - MSE_model/MSE_best_interp`, on clean non-augmented val.
- [ ] **SUCCESS = `CES_TI`: model RMSE strictly lower than the best interpolation baseline with
      non-overlapping 95% CI across >=5 seeds.**
- [ ] `CES_VT`: report model-vs-interpolation honestly; expected to NOT beat it; frame as the
      `T_i`↔`V_rot` asymmetry with physics explanation (NBI torque unobserved; Mirnov aliased).
- [ ] Gap-length stratified results reported (reuse `analyze_gap.py`) so "where it wins" is explicit.
- [ ] MNAR limitation stated: skill is measured on observed points only (optimistic bound).
- [ ] Final fixed model selected via the AutoML loop, reproducible (fixed seed, pinned config);
      loop method documented in appendix.

## Assumptions Exposed & Resolved
| Assumption | Challenge | Resolution |
|------------|-----------|------------|
| Contribution is the physics finding | Round 1 fork | Contribution is **the model** (beats interpolation); physics asymmetry is the accompanying insight |
| "Better" = beat persistence | Round 2 | "Better" = **beat conventional interpolation (linear/spline/AR)** actually used in practice |
| Fair comparison needs causal baselines | Round 3 | Baselines may use **past+future** (offline); beating them is the strong claim |
| Must beat interpolation on both targets | Round 4 (contrarian) | **`CES_TI` win = success**; `CES_VT` non-win = expected asymmetry finding |
| AutoML loop is the deliverable | Round 5 | Loop is the **search method**; thesis reports **one fixed model**; loop in appendix |

## Technical Context
- Current `evaluate.py` compares only against **persistence** — the interpolation baselines
  (linear/spline/AR using past+future CES) are **not yet implemented** and are the top evidence-build task.
- Interpolation baselines read CES values surrounding the target row directly from the CSVs
  (future rows allowed); they are an evaluation-time computation, independent of the model.
- The model already uses fast diagnostics at the target timestep + masked past-CES history;
  contract and shapes are fixed (see `PROJECT_KNOWLEDGE.md` / `CLAUDE.md`).
- AutoML loop now searches architecture family (CNN/Transformer/RNN/SSM) and window size
  (`WINDOW_SIZE` in `model.py`), billed to the Claude subscription via the `claude` CLI.
- Prior ablation (single-seed): `V_rot` predictable from history only; fast diagnostics add ~0
  for `V_rot`, positive for `T_i` — to be re-confirmed multi-seed.

## Ontology (Key Entities)
| Entity | Type | Fields | Relationships |
|--------|------|--------|---------------|
| Nowcasting model | core | fast-diagnostic + past-CES inputs, per-target output | competes with Interpolation baseline |
| Interpolation baseline | core | linear/spline/AR, uses past+future CES | the bar to beat (for T_i) |
| Success metric | core | per-target RMSE/skill, physical units, multi-seed CI | compares Model vs Interpolation |
| T_i / V_rot targets | core | CES_TI (beats interp), CES_VT (asymmetry) | define per-target acceptance |
| AutoML loop | supporting | arch+window search, keep/discard | produces final fixed Model |

## Ontology Convergence
| Round | Entities | New | Stable | Stability |
|-------|----------|-----|--------|-----------|
| 1 | Model, Baseline, Metric, Gap-regime, Target | 5 | - | N/A |
| 2 | + Conventional interpolation, current-practice | 2 | 5 | 71% |
| 3 | + offline interpolation (past+future) | 1 | 7 | 88% |
| 4 | (stable) | 0 | 8 | 100% |
| 5 | + AutoML loop role | 1 | 8 | 89% |

## Interview Transcript
<details>
<summary>Full Q&A (Round 0 + 5 rounds)</summary>

**Round 0 (Topology):** Confirmed 3 components: Contribution claim / Success metric & acceptance / Evidence & scope.

**Round 1 — Contribution claim / Goal (Ambiguity 66%→55%):**
Q: One-sentence core contribution? A: **A better nowcasting model** (performance is the core).

**Round 2 — Success metric / Criteria (55%→42%):**
Q: Define "better" — vs what, where, how much? A: Beat **conventional interpolation (linear AR, spline)** actually used for interpolation today; must find and beat those.

**Round 3 — Evidence & scope / Constraints (42%→29%):**
Q: What do baselines access; offline vs online? A: **Offline — beat interpolation that even uses future CES** (strong claim).

**Round 4 — CONTRARIAN, Success metric / Criteria (29%→20%):**
Q: V_rot likely won't beat future-using interpolation — success/fail? A: **Beating on T_i = success; the asymmetry is the finding.**

**Round 5 — Evidence & scope / Constraints (20%→~12%):**
Q: AutoML loop's place? A: **Loop finds the model; thesis reports one fixed final model** (loop = appendix/reproducibility).
</details>
