# PREREGISTRATION B.28 — does the two-term equation read off the frozen seq_v2 explain the backbone?

Registered 2026-11-04, with code (`b28_equation/run_b28.py`, `b28_equation/stage1_score.py`,
`seq/model_seq_b28.py`, registry entries `b28` / `b28x`), before any stage-1 split other than
the plumbing smoke named in §6 and before any stage-2 run. Named in
`docs/b28_weight_physics_report.md` §6 (the weight/response readout that motivates it).

## 1. Why

승상님 asked whether the trained weights, read top-n, give the layer relations, the dominant
inputs, a short "main equation", and its link to the transport equation. The readout of the
30 frozen seq_v2 checkpoints (`docs/b28_weight_physics_report.md`) answered: weight magnitude
does not (first-layer column norms flat across the 15 fast channels, channel rank Spearman 0.43
across splits, the largest weights on the least influential input MC), but the input–output
response does. It reads as

    y_hat(t) = m(t) + sum_c a_c g(dx_c(t)) + b0

with `m` an exponentially weighted average of the past observed `T_i` (gain ≈ 0.35 at 10 ms,
≈ 0.5 after a 15–55 ms gap; forget-gate memory 1.3 steps in 20/20 checkpoints), `dx_c` the
10 ms increment of fast channel `c` (BES ≈ +50 eV, ECEI ≈ +28 eV per 1σ step, returned within
one step: increment detectors), and `g` saturating (event-row gradient smaller, 16/16). A pilot
on 24 TEST shots per split (not the frozen rows) put the fixed-α EWMA alone at 72–86 % (cut) /
86–94 % (inclusive) of the backbone's persistence-relative skill, EWMA + linear increments at
77–89 % / 89–95 %. B.28 tests that reading where the thesis scores things: on the frozen rows
(stage 1) and end to end under the identical protocol (stage 2).

Consistency obligations: P06 (fast channels = denoising of the anchor, +0.05), P01 (gain vs
PCHIP is event-row transient physics), P03 (memory 17–34 ms), §8bs (photon floor). A stage
result that contradicts one of these must say so in §8bz.

## 2. Design

**Stage 1 (CPU, no training).** For the four cut backbone runs (`.b1_seqv2_s{42,1,7,123}_i42`)
and the four inclusive ones (`.b5i_seqv2_s{s}`), the TEST `CES_TI` row population of
`comparison_errors_test.npz` is re-derived through the harness (`build_clean_val_subset` +
eval_seq's keep / baseline-NaN mask) and verified against the npz (shot ids, `y_true`, `dt_ms`
must match; the script aborts otherwise). On those rows:

- (a) `ewma_α`: EWMA of the past observed `T_i` (physical units, reset at ≥ 0.5 s gaps),
  α ∈ {0.25, 0.35, 0.45}, **no fitted parameter**. α* = the α with the best VAL-file skill vs
  persistence (never chosen on TEST).
- (b) `ewma_inc`: `ewma_α*` + a linear term in the 15 per-shot-z fast-channel 10 ms increments,
  ridge-fitted (λ = 10⁻² n) on the VAL files' observed rows: 16 coefficients.
- (c) the backbone (`se_model` of the npz).

References: the npz's own `se_pchip`, `se_persistence`, `se_gp_causal`, `se_gp`. Skill =
1 − ΣSE_arm/ΣSE_ref; shot-cluster bootstrap CI (B = 10 000, seed 12345, the
`bootstrap_compare.py` resampling); ratio = skill_arm/skill_backbone against the same
reference, with its own shot-cluster CI. Strata reported, never deciding: top-10 % |ΔECEI|
rows vs rest; staleness ≤ 15 / 15–55 / > 55 ms.

**Stage 2 (GPU, 16 runs, one batch at a time).** The equation made trainable:
`CES_SEQ_MODEL=b28` (`seq/model_seq_b28.py`) — `T_i` = EWMA with gain
`K(s) = K_∞ + (K_0 − K_∞) e^{−s/s_K}` of the gap `s` since the previous observation (3
parameters), + Σ_c a_c κ tanh(Δx_c/κ) (15 + 1), + b0 (1): **20 parameters**; `V_rot` branch =
seq_v2's, verbatim, so the ONE controlled variable against the backbone is the `T_i`
computation. `b28x` = the same + one cross term `c · mean(ΔECEI) · mean(ΔBES)` (21 parameters):
the single nonlinear event term tried. Splits {42, 1, 7, 123} × init = split seed × both
populations; env = `GATE_ENV` (W = 2, held-free, per-file cap 500, cut 3000 / 0),
`CES_PER_SHOT_NORM=1`, MSE loss, val-MSE selection, patience 6, epoch cap 60. Control = the
frozen backbone (`.b1_seqv2_s{s}_i{s}`, `.b5i_seqv2_s{s}`), no retraining; harness stats from the
same control metrics as B.1 / B.5; `paired_model_compare` verifies row-for-row pairing;
`bootstrap_compare` gives the PR4 ladder. Learned parameters written to `physics_params.json`
(K_0, K_∞, s_K [ms], a_c [eV per σ], κ, b0).

**Stage 3 (only if stage 2 is negative; registered now as the measurements that would overturn
it).** (i) Jacobian on every frozen row (not a 220-row sample) to read the form of the residual
event term; (ii) context-truncation MSE deficit on the same checkpoints at 1/2/3/5/10 steps, to
join the 0.5/step amplitude decay (readout §2.4) with P03's 0.559/step MSE decay in one number;
(iii) the MC first-layer column-norm trajectory during stage-2 training (the "random walk on a
white input" hypothesis of readout §2.2). All CPU except (iii), which is a log of stage 2.

## 3. Stage-1 decision rules (fixed now)

- **H28-1 (primary).** ratio_b = skill(`ewma_inc`)/skill(backbone) vs persistence ≥ 0.75 on
  **4/4 cut splits** → "the two-term equation carries the backbone's persistence-relative
  skill". Inclusive reported with the same rule, secondary (its event rows include the > 3 keV
  fit-failure spikes the backbone learned to ignore, readout §7.4).
- **H28-2.** skill(`ewma_α*`) vs PCHIP within [−0.05, +0.05] on 4/4 cut splits → "the bulk
  denoising gain cancels against the acausal PCHIP, so the backbone's PCHIP margin is the
  event term" (the reconciliation of the readout with P01). Outside the band on the positive
  side → the causal EWMA beats the acausal interpolation in the bulk as well (foundations
  §9.1: PCHIP amplifies anchor noise), and P01's "90–114 % transient" is then read as a
  statement about the margin over the EWMA, not over persistence; this is recorded, not
  adopted as a claim.
- Reported, never deciding: all three α on TEST, the ratio CIs, the strata, the 16 increment
  coefficients in eV per σ (compared with the readout's impulse sums, BES ≈ +50, ECEI ≈ +28 eV).

## 4. Stage-2 decision rules (fixed now)

- **H28-3 (primary).** ratio_2 = skill_vs_PCHIP(`b28`)/skill_vs_PCHIP(backbone control) ≥ 0.70
  on **4/4 cut splits** → "the equation explains the model". Mean ratio in [0.50, 0.70) →
  "the event-row nonlinear term is needed" (stage 3 runs). Mean < 0.50 → the §4 equation is
  discarded as an explanation of the backbone. Inclusive: same readout, secondary.
- **H28-4.** `b28x − b28` paired `T_i` skill significant (shot-cluster CI excludes 0, arm
  better) on ≥ 3/4 cut splits → the cross term is named as the event term; otherwise null.
- **Guards (reported).** `V_rot`: `b28` vs control paired skill, count of significant deficits
  (the branch is the same architecture retrained; a deficit would be an optimisation artefact,
  not a design result). `T_i`: count of splits where the control is significantly better
  (expected 4/4; it is the quantity H28-3 measures, not a failure).
- **Physics readout.** Σa_BES ∈ [+25, +75] eV/σ and Σa_ECEI ∈ [+10, +45] eV/σ, K_0 ∈ [0.25,
  0.50], on 4/4 cut splits → "the trained equation lands where the readout put the backbone".

No other rule decides a claim. Nothing here changes a registered headline.

## 5. Predictions (written before any stage-1 split other than §6's smoke, before any stage-2 run)

| cell | prediction | confidence |
|---|---|---|
| H28-1 cut 4/4 (ratio_b ≥ 0.75) | supported | 0.75 (pilot range 0.77–0.89 on sampled rows) |
| ratio_b inclusive | 0.85–0.95 on 4/4 | 0.7 |
| H28-2 cut (EWMA vs PCHIP inside ±0.05) | supported | 0.45 (see §6: the smoke split sits at +0.06) |
| `ewma_inc` vs PCHIP, cut | +0.03 … +0.10, 4/4 positive | 0.6 |
| backbone vs PCHIP, cut (record) | +0.17 … +0.27 | 0.95 |
| strata | backbone − `ewma_inc` gap larger on top-10 % rows than on the rest, 4/4 | 0.8 |
| H28-3 cut (ratio_2 ≥ 0.70 on 4/4) | supported | 0.55; mean ratio 0.60–0.80 (0.7) |
| H28-4 cross term | null | 0.6 |
| Σa_BES, Σa_ECEI, K_0 in the stated bands, 4/4 | supported | 0.65 |
| `V_rot` significant deficits (b28 vs control) | 0–1 of 4 | 0.7 |
| stage-2 run time | 5–20 min per run (CPU TorchScript scan, 20 parameters) | 0.7 |

## 6. Disclosures

- The pilot numbers in §1 come from `docs/b28_weight_physics_report.md` §3: 24 TEST shots per
  split, all observed rows, α chosen on TEST there (a leak, fixed here by VAL selection). They
  are not the frozen rows and are not cited as results.
- The plumbing smoke of stage 1 (`stage1_score.py --smoke`: cut s42 only, 30 VAL files, B = 300)
  was run before this document was finalised. It printed ratio_b 0.80, `ewma_α*` vs PCHIP
  +0.061, `ewma_inc` vs PCHIP +0.082, backbone +0.174. The s42 cut cell is therefore not a
  fresh test; the H28-2 band is kept as first written from the pilot (≈ 0) and its confidence
  lowered to 0.45 above. The full stage 1 recomputes s42 with all VAL files and B = 10 000.
- The readout's own skill numbers (persistence-relative, sampled rows) are not comparable to
  the thesis headline; only stage 1's frozen-row numbers are.
- **Erratum 2026-11-08 (before any stage-1 split other than cut s42 was scored).** §2 named the cut
  runs `.b1_seqv2_s{s}_i42`. The headline cut runs, the B.5 rescore and stage 2's own control are
  the seed-matched `.b1_seqv2_s{s}_i{s}` (identical for s = 42); the `_i42` directories for
  s ∈ {1, 7, 123} are 2026-08-14 gate runs whose npz lacks the persistence / GP / y_true columns,
  so stage 1 could not have been scored against them. Stage 1 uses `_i{s}`. For the cut s1 / s7 /
  s123 npz the `_y_true` column is present and verified; the rule, the arms and the predictions are
  unchanged.
- **Stage-3 addendum 2026-11-09 (written before any stage-3 measurement other than the plumbing
  smokes named below).** Stage 2 did not log the first-layer column norms (`run_b28.py` had no such
  log), so (iii) is taken from ONE seq_v2 re-train of the cut split 42 / init 42 under the B.1 env
  with `CES_SEQ_LOG_WIH=1` (per-epoch W_ih column norms, `train_seq.py`; the flag is off by default
  and touches no optimisation). (ii) reuses the §8ac artifacts `.reach_s{s}` for the cut population
  (same frozen checkpoints; identity re-verified on shot / dt_ms / se_pchip / se_model) and scores
  the inclusive checkpoints with `CES_SEQ_CONTEXTS=1,2,3,5,10,20,50` on CPU. Reading criteria,
  fixed now, reported not deciding (stage 3 carries no claim): (i) a candidate residual family
  whose cross-fitted (shot-parity) fit closes ≥ 50 % of the backbone − equation PCHIP-skill gap on
  4/4 cut splits names the form of the event term; the lag-0 gain's saturation form is read from
  the sech² / Cauchy / flat SSE on all rows × channels, and its state dependence from the partial
  R² of |Δx_{t−1}|, staleness and |carried − EWMA|. (ii) the per-step decay of the truncation
  deficit ρ_MSE (log-linear on k ≤ 10) "joins" the amplitude decay ρ_amp (carried-T_i gradient,
  lags 0–3, all rows) if ρ_MSE ∈ [ρ_amp², ρ_amp] on 4/4 cut splits; P03's 0.559 is compared with
  the same fit on k ∈ {2, 3, 5}. (iii) the random-walk reading is supported if the MC columns'
  norm² − norm₀² grows linearly in optimizer steps (R² ≥ 0.9, growth not saturating) with a slope
  ≥ 2× the BES columns'; it is rejected if the MC norm shrinks or grows no faster than BES.
  Plumbing smokes run before this addendum was final: none (the stage-3 script was not executed).

## 7. What would change the paper

- H28-1 and H28-3 supported → one paragraph in the model section: "what the backbone computes"
  (the equation, its parameters in physical units, and the share of skill it carries), closing
  §8ba/§8bb's physics narrative from the model side. No headline number changes.
- H28-3 not supported → the paragraph states the share the equation carries and that the
  remainder is an event-row term the weights do not expose; stage 3 is run and recorded.
- Either way the `V_rot` statements are untouched (the branch is not varied).

## 8. Not in scope

No change to the backbone, the loss, the populations, the splits or the harness; no W = 2
window control (the pairing is against the seq_v2 backbone); no GPU use in stage 1.

## 9. Stage 4 (registered 2026-11-09, after §8bz-3, before any `b28h` run): does the backbone's own kernel recover the bar?

**Why.** Stage 3 (§8bz-3) read two form errors off the backbone that the §2 equation does not have: the
fast-channel term is x_t minus the preceding ≈ 10-step mean (a ≈ 100 ms high-pass), not the 10 ms
increment, and the gain is flat to 2σ and switched off beyond 3σ, not a κ ≈ 1σ tanh. Stage 4 asks
whether correcting the *form* alone, with the same 20 parameters, recovers the registered 0.70 bar.

**Design.** `CES_SEQ_MODEL=b28h` = `SeqCESB28(fast_term="hp", nonlin="reject")`: T_i = EWMA m (gains
K_0, K_∞, s_K as before) + Σ_c a_c · d_c · σ((θ − |d_c|)/0.5) + b_0, with d_c = x_c,t − mean(x_c,t−10..t−1)
(fewer rows at a block start; 0 at t = 0) and θ learned (softplus, init 3σ): 3 + 15 + 1 + 1 = **20 T_i
parameters**; V_rot branch = seq_v2's, verbatim. Env, splits, controls, selection and pairing exactly
as stage 2 (§2); 8 GPU runs (cut + inclusive × 4 splits), one batch at a time. Control = the frozen
backbone; second pairing = the stage-2 `b28` run of the same split (`paired_b28h_vs_b28.json`).
Summary `data/.b28_stage4.json` (`run_b28.py --stage 4`).

**Rules (fixed now).**
- **H28-5 (primary).** ratio vs PCHIP = skill(`b28h`)/skill(backbone) ≥ 0.70 on **4/4 cut** → "the equation
  with the backbone's kernel explains the model" (the §7 paragraph is then rewritten with the corrected
  form). Mean in [0.60, 0.70) → "form corrections help but a nonlinear remainder stays". Mean < 0.60
  (stage 2's value) → "the form corrections are not the gap". Inclusive: same readout, secondary.
- **H28-6.** `b28h − b28` paired T_i skill significant (shot-cluster CI > 0, arm better) on ≥ 3/4 cut →
  the high-pass + rejection form is named the better description of the backbone; otherwise null.
- **Guards (reported).** V_rot deficits vs control (expected 0–1 of 4). θ in [2, 5]σ. Σa_BES in [+40, +80],
  Σa_ECEI in [+20, +50] eV/σ (the §8bz-3 Jacobian impulse bands, cut). K_0, K_∞, s_K within ±0.05 / ±0.05
  / ±10 ms of stage 2's.

**Predictions.**

| cell | prediction | confidence |
|---|---|---|
| H28-5 cut 4/4 (ratio ≥ 0.70) | supported | 0.40 |
| mean ratio vs PCHIP, cut | 0.64–0.74 | 0.6 |
| H28-6 (b28h > b28 significant ≥ 3/4 cut) | supported | 0.75 |
| θ, cut | 2.5–4.0σ on 4/4 | 0.6 |
| Σa_BES, Σa_ECEI in the impulse bands, 4/4 cut | supported | 0.6 |
| V_rot significant deficits | 0–1 of 4 | 0.7 |
| run time | 5–20 min per run | 0.7 |

**Disclosures.** The kernel and the rejection form were read from the stage-3 Jacobian on the same
frozen checkpoints and the same TEST rows that stage 4 is scored on; stage 4 is therefore a test of
whether that reading transfers to end-to-end training, not an independent discovery of the form.
The baseline length (10 steps) and the rejection width (0.5σ) are fixed from §8bz-3, not fitted.
Plumbing smoke before this section was final: none (`--stage 4 --smoke` is run after it is written).
