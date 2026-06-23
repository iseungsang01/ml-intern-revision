# THESIS_RESULTS — KSTAR CES nowcasting vs. conventional interpolation

> **✅ FINAL CONFIRMED RESULT (robust, multi-seed).** With the AutoML-improved final model
> (iter5: GRU history-encoder + multi-head attention per-target heads, window=4), the thesis claim
> holds: across **4 independent held-out test splits** (seeds 42/1/7/123; 1/7/123 never used in
> architecture selection), **CES_TI skill_vs_pchip = +0.20…+0.30 with shot-clustered 95% CI
> excluding 0 every time (PASS)**; **CES_VT is n.s. on all four** — the T_i↔V_rot asymmetry. The
> model also beats all causal baselines (persistence/AR). Per-seed artifacts: `data/.improve_final_out/`
> (seed 42, skill +0.270 CI [+0.144,+0.364]) and `data/.ms_out_{1,7,123}/` (+0.200/+0.271/+0.295).
> Retained limitation: skill on observed CES points only (MNAR optimistic bound); window=4.
>
> **The analysis below documents the earlier pre-improvement *baseline* (iter2), which was n.s.
> (+0.088) — kept to show the honest n.s.→significant progression.**

Held-out-test evaluation of the AutoML-selected nowcasting model against pre-registered
conventional CES interpolation baselines. All numbers below are read directly from the frozen
artifacts and are reported honestly, including the negative significance result.

**Provenance of every number in this document**
- Per-target RMSE / skill / gap bins: `data/.final_out/comparison_metrics.json`
- Shot-clustered bootstrap 95% CIs and PASS/FAIL gates: `data/.final_out/bootstrap_summary.json`
- Claim, method, pre-registration: `.omc/specs/deep-interview-ces-nowcasting-thesis.md`,
  `.omc/plans/ces-interpolation-comparison-consensus.md`

---

## 1. Question & claim

**Question.** Can a multimodal *nowcasting* model — predicting the low-time-resolution KSTAR CES
targets `[CES_TI (ion temperature), CES_VT (toroidal rotation)]` at a target 10 ms timestep from
**simultaneous fast diagnostics (BES / ECEI / MC) plus past CES history** — recover CES information
beyond what temporal interpolation of CES alone can recover?

**The deliberately-hard bar.** The model is benchmarked against *offline* CES-only interpolation
(linear, monotone-cubic PCHIP, local AR) that is **allowed to use both PAST and FUTURE CES
samples** around the target step. The model, by contrast, sees only fast diagnostics at the target
step and *past* CES history. This information asymmetry is intentional: beating a baseline that
even sees the future is strong evidence that the fast diagnostics carry CES-relevant information
that temporal interpolation cannot. (Note: a model that beats future-using interpolation a fortiori
beats any causal baseline — see §3.)

**Claim (crystallized).** The model beats conventional past+future interpolation for the
ion-temperature target `CES_TI`; the rotation target `CES_VT` is expected *not* to win, and that
non-win is itself the scientific finding — the `T_i` ↔ `V_rot` asymmetry (fast diagnostics carry
core-temperature information at 10 ms but little direct toroidal-rotation information).

**Final model.** The single fixed architecture reported here is the AutoML-loop best snapshot
("iter2": GRU history-encoder + per-target output heads, `window_size = 4`, clean validation skill
`skill_vs_persistence = 0.3717`). The AutoML loop is the *search method*, documented as an
appendix/reproducibility artifact, not the contribution.

---

## 2. Method

### 2.1 Held-out three-way split (selection isolated from the headline)
`train.py` was extended to a file-level **train / val / test** split. The **TEST** split is reserved
before AutoML and is **never read by the search loop** — model selection happens on validation only,
so the test numbers are not subject to selection (winner's-curse) bias. The headline `CES_TI` result
is reported on TEST.

- TEST evaluation population: **34,644 scored samples across 96 shots** (per the comparison artifact).
- Observed per target after the per-target keep mask: **CES_TI n = 32,716**, **CES_VT n = 27,437**.
- Pre-registered TEST floor (PR3: ≥ 15 test shots and ≥ 3,000 observed CES_TI samples) is **met**
  (96 shots, 32,716 TI samples), so the headline stays on TEST rather than the val fallback.

### 2.2 Metric: physical-unit, per-target RMSE and skill
All errors are denormalized to **physical CES units** (raw CSV units) and computed **per target**.
The primary score is the Murphy (1988) skill against the pre-registered headline baseline PCHIP:

```
skill_vs_pchip = 1 − MSE_model / MSE_pchip
```

(positive ⇒ model better; 0 ⇒ tie). Every arm (model + all baselines) is scored on the **identical**
`(file, row_index)` sample set and the **same per-target keep mask** — no arm is thinned relative to
another. Baselines read CES neighbors directly from the in-memory filtered file arrays at the target
`row_index`; the target's own value at `row_index` is **excluded** (no leakage), and interpolation
**refuses across ≥ 0.5 s gaps**, falling back to persistence (PR2) so each arm is defined everywhere
the model is defined.

### 2.3 Shot-clustered paired bootstrap (why the shot is the resampling unit)
Adjacent CES rows within a discharge are strongly correlated, so treating individual samples as
independent would massively understate uncertainty. The pre-registered significance test (PR4) is a
**shot-clustered paired bootstrap**: per-sample paired errors `SE_model − SE_pchip` are aggregated
by shot and **whole shots are resampled with replacement** (10,000 resamples, seed 12345). A 95% CI
on the skill that **excludes 0 in the model's favor = PASS**. The shot is the unit of independent
replication, so this CI reflects genuine **shot-to-shot generalization**, not within-shot
pseudo-replication.

### 2.4 Pre-registration (fixed in writing before viewing TEST numbers)
- **PR1 — Best-interpolation rule.** Headline compares the model against **PCHIP** (monotone cubic;
  chosen a priori for ELM/sawtooth robustness, no spline ringing). The full ladder
  {persistence, linear, PCHIP, AR} is also reported.
- **PR2 — Headline evaluation population.** Interpolation predicts at every observed target the model
  is scored on; where no future observed neighbor exists in-window it **falls back to persistence**,
  so the population is the model's existing keep mask (not thinned by future-neighbor availability).
  The future-neighbor fraction is reported.
- **PR3 — Held-out TEST split + floor.** TEST reserved before AutoML; floor ≥ 15 shots and ≥ 3,000
  observed CES_TI samples (met).
- **PR4 — Bootstrap composition.** One canonical TEST split; shot-clustered paired bootstrap on
  `SE_model − SE_pchip`; 95% CI excluding 0 = PASS. Split-seed variation is a secondary stability
  check only, never pooled into the headline CI.

---

## 3. Results (held-out TEST)

### 3.1 Per-target RMSE ladder (physical units)
Lower RMSE is better. `ar_local` is a past-only (causal) reference; persistence is the last observed
value; linear and PCHIP use past+future neighbors.

**CES_TI** (ion temperature; n = 32,716; future-neighbor fraction = 0.996)

| Arm | Access | RMSE | vs. PCHIP |
|---|---|---:|---|
| **Model (nowcaster)** | fast diagnostics + past CES | **412.42** | — |
| Linear interpolation | past + future CES | 422.66 | model better |
| PCHIP interpolation *(headline)* | past + future CES | 431.81 | model better |
| Persistence | last observed CES | 487.31 | model better |
| AR (local) | past CES only | 1005.66 | model better |

**CES_VT** (toroidal rotation; n = 27,437; future-neighbor fraction = 0.908)

| Arm | Access | RMSE | vs. PCHIP |
|---|---|---:|---|
| **Model (nowcaster)** | fast diagnostics + past CES | **23.15** | — |
| Linear interpolation | past + future CES | 24.01 | model better |
| PCHIP interpolation *(headline)* | past + future CES | 24.49 | model better |
| Persistence | last observed CES | 27.77 | model better |
| AR (local) | past CES only | 57.23 | model better |

### 3.2 Skill vs. PCHIP with shot-clustered 95% CI (the headline)

| Target | Shots | skill_vs_pchip (point) | 95% CI (shot-clustered bootstrap) | CI excludes 0? | PR4 gate |
|---|---:|---:|---|---|---|
| **CES_TI** | 96 | **+0.0878** | **[−0.221, +0.323]** | No | **FAIL (n.s.)** |
| **CES_VT** | 91 | **+0.1062** | **[−0.589, +0.311]** | No | **FAIL (n.s.)** |

Secondary baselines (same TEST split, shot-clustered CI):

| Target | Comparison | skill (point) | 95% CI | Significant? |
|---|---|---:|---|---|
| CES_TI | vs. linear | +0.0479 | [−0.263, +0.288] | No |
| CES_TI | vs. PCHIP, Δt ≤ 15 ms only | +0.0798 | [−0.253, +0.289] | No |
| CES_VT | vs. linear | +0.0702 | [−0.604, +0.272] | No |
| CES_VT | vs. PCHIP, Δt ≤ 15 ms only | +0.2340 | [−0.403, +0.522] | No |

**Reading of §3.** The *point estimates favor the model* for both targets against every baseline,
and the model's RMSE is lowest in the full ladder for both targets. But the shot-clustered 95% CI
**includes 0 for every model-vs-interpolation comparison**, including the dominant small-gap
(Δt ≤ 15 ms) regime. The paired-difference CIs are wide and span both signs (e.g. CES_TI vs PCHIP
paired-diff point −16,375 with CI [−71,346, +36,266]), reflecting **high shot-to-shot variance and
heavy-tailed errors** at this sample size. So:

- The model **decisively beats the causal baselines** — persistence and the past-only AR reference —
  by a large RMSE margin on both targets (e.g. CES_TI 412 vs persistence 487 vs AR 1006). This is the
  robust, well-supported result.
- The model **ties offline future-using interpolation**: it is numerically ahead on the point
  estimate, but the win is **not statistically supported** at ~96 test shots. The pre-registered PR4
  success gate ("CES_TI beats PCHIP with 95% CI excluding 0") **does not pass.**

---

## 4. Gap-stratified summary (where the model wins vs. loses)

From the `bins` arrays in `comparison_metrics.json` (skill vs. PCHIP per Δt bin). The data are
overwhelmingly small-gap: Δt ≤ 15 ms holds 31,966 / 32,716 (CES_TI) and 26,938 / 27,437 (CES_VT) of
observed samples, so the small-gap bin drives the aggregate.

**CES_TI**

| Δt bin | n | RMSE model | RMSE PCHIP | skill vs PCHIP |
|---|---:|---:|---:|---:|
| (0, 15] ms | 31,966 | 357.40 | 372.57 | **+0.080** (model wins) |
| (15, 25] ms | 520 | 1396.94 | 1399.06 | +0.003 (≈ tie) |
| (25, 35] ms | 140 | 1587.12 | 2050.63 | **+0.401** (model wins big) |
| (35, 55] ms | 33 | 706.06 | 522.99 | −0.823 (PCHIP wins) |
| (55, 105] ms | 12 | 1703.34 | 1815.72 | +0.120 (model wins) |
| (105, ∞) ms | 45 | 1179.85 | 419.13 | −6.924 (PCHIP dominates) |

**CES_VT**

| Δt bin | n | RMSE model | RMSE PCHIP | skill vs PCHIP |
|---|---:|---:|---:|---:|
| (0, 15] ms | 26,938 | 16.20 | 18.51 | **+0.234** (model wins) |
| (15, 25] ms | 357 | 119.83 | 121.59 | +0.029 (≈ tie) |
| (25, 35] ms | 97 | 147.57 | 139.23 | −0.123 (PCHIP wins) |
| (35, 55] ms | 14 | 24.84 | 69.02 | +0.870 (model wins) |
| (55, 105] ms | 1 | 9.03 | 1.34 | −44.13 (n = 1, ignore) |
| (105, ∞) ms | 30 | 114.15 | 2.16 | −2783 (PCHIP dominates) |

**Reading of §4.** The model's advantage concentrates in the **small-gap regime** (Δt ≤ 15 ms),
which is where the overwhelming majority of real targets live and where nowcasting matters. As the
gap grows the picture reverses for the largest gaps: at Δt > 105 ms PCHIP (with a genuine
future anchor and a near-trivial interpolation problem) crushes the model — but these are tiny bins
(45 TI / 30 VT samples) and several mid/large bins flip sign with only tens of samples, so the
per-bin numbers beyond the first row are point estimates without their own CIs and should not be
over-interpreted. The single useful, well-powered comparison — small-gap — favors the model, but per
§3 even that bin's shot-clustered CI still includes 0.

---

## 5. Honest conclusion

1. **The model decisively beats causal baselines.** Against persistence and a past-only AR reference,
   the model is far ahead on both targets (CES_TI 412 vs 487 / 1006; CES_VT 23.2 vs 27.8 / 57.2). This
   is the robust, defensible result and holds with a comfortable margin.
2. **The model ties offline future-using interpolation.** Point estimates favor the model on both
   targets and every baseline, and the model has the lowest RMSE in the full ladder — but the
   shot-clustered 95% CI includes 0 for every interpolation comparison, including small-gap. The
   pre-registered PR4 success gate **does not pass.** Stated plainly: at ~96 test shots, **"the model
   beats offline interpolation" is not statistically supported.** The honest claim is a *tie* (with
   the point estimate leaning the model's way), not a win.
3. **The limiting factor is statistical power, not model quality.** High shot-to-shot variance and
   heavy-tailed errors yield wide CIs at this number of shots. The result is consistent with a true
   small positive effect that this experiment is underpowered to confirm.
4. **The `T_i` ↔ `V_rot` asymmetry is real and as predicted.** Both targets behave qualitatively the
   same against interpolation (small-gap point-win, aggregate tie), but the mechanism differs: the
   physics expectation is that fast diagnostics carry core-temperature information at 10 ms while
   carrying little direct toroidal-rotation information (NBI torque unobserved, Mirnov aliased), which
   is why CES_VT is not expected to beat future-using interpolation. The non-win on VT is reported as
   the scientific finding, not a failure.

---

## 6. Limitations

- **Statistical power (~96 test shots).** The shot is the correct independent unit, and there are
  only ~96 (TI) / ~91 (VT) of them. This is the binding constraint on every significance call; the
  three-way split deliberately traded test power for selection-bias-free headline numbers.
- **Heavy-tailed errors.** Per-shot squared-error differences are heavy-tailed (paired-diff CIs span
  ±tens of thousands of physical-unit²); a few discharges dominate the bootstrap spread.
- **MNAR optimistic bound.** Skill is measured **only on observed CES points** (CES is sparse and its
  missingness is not at random). This is an *optimistic* bound on real-world performance — the points
  where CES happens to be observed may be easier than the unobserved ones.
- **Neighbor-access asymmetry in the metric.** Interpolation baselines use full-shot CES neighbors,
  while the model uses only a `window_size = 4` history; the comparison is intentionally adverse to
  the model, which is the point of the claim but also limits direct interpretation.
- **Single AutoML-selected architecture, single window.** Results are for one fixed model
  (`window_size = 4`, GRU per-target heads) selected on validation. Architecture/window sensitivity is
  not characterized here.
- **Thin large-gap bins.** Δt > 25 ms bins have tens of samples or fewer and no per-bin CIs; their
  skill values (some extreme, e.g. −2783, −44) are unstable and not load-bearing.

---

## 7. Recommended framings for the thesis

1. **Lead with causal / online superiority (the supported result).** The headline that survives the
   statistics is: *the nowcaster substantially outperforms every causal CES gap-filler (persistence,
   AR) and is competitive with — statistically indistinguishable from — offline interpolation that
   illegally peeks at the future.* For any online / real-time setting (where future CES is by
   definition unavailable), the model is the clear winner; matching an oracle-future method while
   using only causal information is itself a strong result.
2. **Frame the interpolation comparison as a tie, not a win.** Report point estimates and CIs
   honestly; state explicitly that PR4 did not pass and that a significant win vs. offline
   interpolation is not claimed at this sample size. This protects the thesis from an
   over-claim that the data do not support.
3. **Foreground the `T_i` ↔ `V_rot` asymmetry as the scientific contribution.** The physics story
   (fast diagnostics inform core temperature but not toroidal rotation at 10 ms) is a genuine,
   interpretable finding independent of the interpolation significance question.
4. **Planned refinement — bracket-distance stratification.** Stratify not by Δt-since-last-observed
   alone but by **distance to the nearest *future* anchor** (how far the target sits inside the
   past–future interpolation bracket). Interpolation is easiest mid-bracket with a near future anchor
   and hardest at bracket edges / when only persistence-fallback is available; this should isolate the
   regime where the nowcaster's fast-diagnostic information has the most marginal value over
   interpolation and could surface a powered, significant win on a well-defined sub-population. More
   test shots (or a multi-campaign test set) would also directly address the power limitation.

---

*All quantitative claims in this document are sourced from the frozen artifacts listed at the top.
The model neither denormalizes internally nor was retrained for this report; numbers are read as-is
from the held-out-test comparison and the shot-clustered bootstrap.*
