# Four research programs for the pedestal-top CES nowcast: status, retractions, ranking

I fitted or ran nothing here; all proposals are designs to register before any fit or readout. Tags: [S] source fact, [A] paper-author claim, [H] my application hypothesis, [U] unknown. Code locators are path::function only, since the blobs were unnumbered and I assert no line numbers.

## 1. Status ledger

| Item | Status |
|---|---|
| Hold, causal GP, two-sided PCHIP+GP; GP+boosted history/fast (B73/B75); pooled fit, raw windows, TXrel (B84); controls (B79/81/82); latency curve (B87) | Completed current-line readouts. Rows and fits are not guaranteed identical across controls. |
| TCN fast/history pair (B94) | Evaluated on outer TEST: FULL15 one seed and FOUR three seeds, kept separate. |
| LSTM encoders A (anchor) and C (no-anchor), TCN T (B92) | Pilot only: four units, one seed, validation gates, 4–12 epochs, mostly budget-limited. This is not a converged architecture comparison. |
| B90 G/T/R/C/U/V/N (decay GRU, both diagonal observers, Huber, Student-t, Gaussian mean/variance) | Registered and implemented, then superseded by B92. They were never evaluated in the completed A/C/T pilot or on TEST, so no empirical verdict exists. |
| B88 anchor encoder | Implemented; evaluated only as pilot candidate A. |
| B98 greybox | TRAIN/internal VAL, F only, four units, cluster overlap. Not TEST evidence. |
| B97 | Generic selector proof only. |
| B104 | Registered, not a result. It owns the budget question. |

## 2. Source facts that retract earlier rankings

1. **"Observers and robust losses already failed."** Retract. They were deferred on cost and relevance (PREREGISTRATION_B92). Any ranking that demoted filters or likelihood heads on that basis is void.
2. **"The existing observer has a fixed gain; a learned gain is the differentiator" (dossier 02 §5).** Retract. `models90.py::DiagonalObserver._step` already propagates variance and computes a gain from input-dependent noise on every row. The real differences are event semantics, source-time replay and stream factorisation.
3. **"The structure forbids any update without a new CES sample."** Retract as too strong. Fast inputs may drive the state every step; only the CES update is event-gated.
4. **"RKN handles missing inputs."** Qualify. The official cell masks only when `never_invalid` is off (`RKNCell.py::RKNCell.forward`), whereas CRU masks unconditionally (`CRUCell.py::RKNCell._update`) [S].
5. **"Learned filters need known f/h."** Retract. That applies to original KalmanNet only (PDF p.3 §II-B; `KalmanNet_nn.py::KNet_step`) [S].
6. **"A new local-level Kalman comparator is needed" (dossier 02 §4).** Downgrade. `model98.py::matrix` already implements strictly-earlier-source exponential propagation, so reuse it rather than refit a duplicate.
7. **"Anchor-dropout training is a new intervention" (dossier 01 H1).** Retract. Mixed availability already underlies B85/B92/B94 and B84. A monotone age response is not implied.
8. **"State information implies the L5 increment is at least the F increment" (H2), and "window features beating snapshots shows aliasing; the subset is underpowered" (H3).** Retract all three inferences. Precision of the subset is unknown.
9. **"No discharge-level holdout or deployed gap inference in the lineage."** Retract. COMPASS splits by discharge (PDF p.6) and the Diag2Diag script splits by discharge index, with the caveat that it runs on dummy data. RTCAKENN ran in the PCS without valid TS (§4.2.1) [S]. Our validation and non-deployment are therefore not distinctive in either direction.
10. **"causal=True makes TS2Vec causal."** Retract. `ts2vec.py::encode` defaults to causal=False and sliding_length=None, and only the sliding branch restricts right context [S].
11. **"B90-N covers reliability."** Retract. A joint Gaussian NLL changes the mean; the faithful loss (Stirn Eq. 5, p.4) does not.

## 3. What the audited packet implies for program choice

I assume a positive paired value means the fast-enabled TX beats the history twin TH. This sign convention is a prerequisite check.

- **The contrast is design- and rule-dependent.** In FULL15, V1 D1000/F is +0.256 [0.198, 0.309], V3 D10/F is −0.518 [−0.717, −0.348], and V2 D10/L5 is −0.765 [−1.33, −0.35]. The FOUR three-seed subset repeats the V3 D10 sign (−0.292 [−0.533, −0.141]). V4a is mostly indeterminate.
- **These intervals are frozen-fit intervals** that omit fit, selection and multiplicity uncertainty. The D rules are source-age rules, not horizons.
- **The cause is not identified.** Fresh-anchor interference, non-transfer of raw-voltage level across split types, and under-training are all live. Of the 46 confirm fits, 38 were budget-limited.
- **B98 points the same way, weakly.** On internal VAL the fast+NBI forced model beats its history twin by at most about 3% with intervals often spanning zero. The static anchor-difference comparator L has lower MAE in nearly every cell.
- **Compute.** TX p99 is 4.56 ms on one CPU thread, excluding acquisition and transfer.

Under V1–V4a treated equally, the sharpest open question is why unconstrained fusion is neutral to harmful where the anchor is fresh, and whether a structural constraint removes that without erasing the stale-anchor increment.

## 4. Program A: source-time, event-gated anchor-residual estimator

**Competing explanations.**
- E-contract: the short-age deficit comes from treating held anchors and fast inputs as per-row evidence.
- E-level: it comes from absolute raw-voltage level that does not transfer.
- E-none: fast signals carry no report-relevant increment beyond NBI and slow context.

**Standard components (no novelty claimed).** CRU-style masked update, textbook delayed-measurement replay (no primary source supplied, so none cited), the B88 shared embedding, the B98 source index. The closest fusion analogues are RTCAKENN's last-valid hold (PDF p.6) and Diag2Diag's forward-filled CER (`reindex(..., method='ffill')`). Neither uses anchor age or event updates.

**Minimum model.** Let s(t) be the latest visible, strictly earlier source index, a_t its age and b_t the frozen causal-GP mean.

- A1, residual: ŷ_t = b_t + γ(a_t, flag)·uᵀ[φ(x_t) − φ(x_s)], with γ(0)=0 and γ bounded and monotone, for example 1 − exp(−a/τ_γ). The explicit difference cancels channel offsets. Gain drift is not cancelled.
- A2, filter:
  - Propagation: z⁻_t = α⊙z_{t−1} + (1−α)⊙d(x_t) and P⁻_t = α²P_{t−1} + q(1−α²). Fast input is a drive and does not shrink variance.
  - Decode ŷ_t before any use of F_t.
  - When a report y_s first becomes visible: ν = y_s − ŷ_s, using the stored pre-update decode. K = P⁻_s h/(hᵀP⁻_s h + r(flag)), then z_t ← z_t + α^{t−s}⊙Kν, with the matching variance reduction.

**Controls (same 22 slots, same rows, matched update counts rather than wall-clock).**
1. An every-row-update twin, which is the B90 observer contract.
2. A concatenation twin (B88-A) against the difference form.
3. A zero-fast history twin.
4. B98 M/H as the linear source-time reference, within its scope.
5. The existing permutation and NBI-only input controls.

**New information.** Whether event semantics and differencing change the sign pattern at short source age, per design. No existing run isolates this. B90 R/C are every-row models and were unevaluated.

**Data.** Feasible on the 10 ms grid with nominal stamps. Real arrival times are [U], so the replay is a constructed scenario.

**Decisions.** Dimensions are fixed a priori. Stopping and any hyperparameter are chosen on internal VAL. One TEST readout is taken on FOUR three-seed across all four designs for F and L5, with paired shot-cluster differences and a registered margin. The motivation is TEST-informed, so the result is exploratory.

**Falsifiers.**
- If A2 minus the every-row twin fails the margin, E-contract is unsupported.
- If the difference form minus the concatenation form fails, E-level is unsupported.
- If the increment over the history twin persists under permuted fast inputs or is matched by NBI, it is not a fast-diagnostic increment.
- An interval spanning both the margin and zero is inconclusive, not equivalence.

**Compute and stop.** A1 is CPU-feasible. A2 is elementwise [A: RKN §2.4], but its 10 ms fit is untimed. The bound is five arms × four units × three seeds, with no architecture search and no FULL15 extension unless the margin is met.

## 5. Program B: report-level reliability (faithful variance head)

**Competing explanation.** Predictive dispersion of the report is either anchor staleness alone or is additionally predicted by fast and raw inputs.

**Components.** Stirn Eq. 5, with Theorem 1 (p.4) making the mean trajectory identical to the MSE twin. β-NLL (Seitzer Eq. 7) is excluded because it moves the mean and would reopen every hold/GP comparison.

**Model.** L = ½(y−μ)² − log N(y; ⌊μ⌋, σ²(⌊trunk⌋)), fitted on the selected TCN.

**Gate.** Point predictions must equal the twin's to numerical tolerance. Two conditions follow that the dossier understated [H]:
- Stopping must be by update count. The current 1200 s wall-clock cap would break the identity, because the extra head changes updates per second.
- There must be no global-norm clipping across heads.

**Controls.** B0 is constant TRAIN-residual variance. B1 is variance tabulated by age × flag, and it is the primary contrast. A separate-variance-network arm is allowed (Stirn p.3). Rows are common-support.

**Readout.** Fixed-quantile interval scores and stratum coverage for F, with L5 shown side by side and not differenced. Shot-level pairing, distribution-free coverage. Serial dependence makes likelihood and ECE descriptive only.

**New information.** The first report-level uncertainty statement tied to the hold/GP framing, with zero effect on the mean claims.

**Data.** Sufficient for report dispersion. Calling σ̂ fit noise would need per-frame χ² or refits, as in the dossier 03 §5 audit. Bishop (PPCF 35, pp.765–768) motivates the non-convergence and MNAR concern for JET, not KSTAR.

**Falsifier.** Faithful minus B1 below the margin means σ̂ is only age bookkeeping.

**Compute and stop.** One run per seed on FOUR, on the current hardware class.

## 6. Program C: monitoring scenario (causal fallback selector)

**Scenario the labels support.** At each grid step, decide from past-only metadata whether to issue the fast-enabled nowcast or fall back to the GP/hold. Evaluate on held-out reports under the existing constructed stale/sparse schedules. This is not a live-delay demonstration.

**Competing explanation.** The harm from fast inputs is either predictable from age, flag and availability, or is idiosyncratic.

**Model.** ŷ = d_t·ŷ_TX + (1−d_t)·ŷ_GP with d_t = 1[g(a_t, flag, avail) > θ]. g is a monotone one-dimensional or small tabular rule fitted on out-of-fold TRAIN predictions, with θ set on VAL.

**Controls.** Age-only threshold, random selector at matched rate, always-TX, always-GP, and a hindsight oracle labelled non-causal.

**New information.** A risk–coverage curve per design that turns the mixed-sign table into an operational statement.

**Prerequisite.** Stored predictions with exact row identities must exist [U]. If they do not, the program needs refits and loses its cost advantage.

**Limits.** The idea is itself TEST-informed, and a selector tuned to the observed D pattern will look good on the same TEST. B97 does not verify it.

**Falsifier.** The selector fails to beat both always-arms by the margin in the designs where signs differ.

**Compute and stop.** CPU only. One rule family, one readout.

## 7. Program D: causal pretraining or privileged teacher

**Competing explanations.** The fast increment is limited by representation and label scarcity, or by information.

**Components.** The TS2Vec objective (PDF pp.3–4) and generalised distillation (GD Eq. 4, p.3).

**Design.** As in dossier 04:
- R0–R4 for pretraining, with R4 the step-matched scratch control.
- T0–T4 for the teacher, with T2 self-distillation, T3 the PCHIP+GP target and T4 permuted teacher outputs.
- A prefix-invariance test on the deployed path.
- A teacher gate on internal VAL before any student fit.

**Obstacles.**
- The population identity E[t|P] = E[y|P] limits any teacher gain to finite-sample variance.
- GD's own evidence shows no gain when the teacher is applied only to labelled points (Exp. 6, p.8), and the share of unlabelled grid points here is [U].
- L5 and B87 already inject future CES. Only future fast signals would be new.
- With 38 of 46 fits budget-limited, pretraining cannot be separated from extra optimisation until B104 reads out.
- It needs GPU and about ten arms.

**Falsifiers.** R1 not better than R4, or T1 not better than T2 or T3.

## 8. Decision matrix

| Rank | Program | Question answered | Uses existing assets | Compute | Main risk |
|---|---|---|---|---|---|
| 1 | A: event-gated anchor residual | Why fusion fails at fresh anchors; whether structure fixes it | B88 φ, B98 source index, B90 as twin | CPU for A1; small GPU for A2 | Motivation is TEST-informed; nominal stamps |
| 2 | B: faithful reliability | Whether dispersion is predictable beyond age | Selected TCN | One run per seed | Cannot be called noise; identity gate is fragile |
| 3 | C: fallback selector | Whether the mixed signs are usable operationally | Stored predictions, if present | CPU | Circularity on reused TEST |
| 4 | D: pretraining or teacher | Whether representation is the bottleneck | TCN | GPU, many arms | Confounded with budget; weak prior from GD |

**Prerequisite audits, not contributions:**
- The sign convention and row identities of stored predictions.
- The share of unlabelled grid points and the B87 values.
- The CES timing and missingness provenance table, built with the diagnostic owners.

## 9. Best estimator-family hypothesis

An anchor-referenced residual estimator, in which the fast contribution is a function of the change in fast-signal embedding since the source index and is gated to vanish at zero source age, with CES entering only as a source-time event. The hypothesis is that it retains the long-age increment of the unconstrained TCN while removing the short-age deficit in V2/V3, relative to its own history twin on identical rows. This is a familiar construction. It promises neither algorithmic novelty nor a physics mechanism, and excludes neither.

## 10. Rejected program

I reject Program D's privileged teacher for now. Its expected information is lowest for the obstacles listed in section 7, and it has the highest arm count. This is not a claim that it cannot work.

## 11. Strongest objection to Program A

The constraint γ(0)=0 partly guarantees the headline result. A model forced toward the GP at short age will show a smaller short-age deficit by construction, so "removing the harm" may be shrinkage rather than better use of fast information. The same shrinkage could be had by the Program C selector at far lower cost.

The claim therefore has to rest on the long-age side: the difference form must retain the increment against the concatenation twin and the permutation and NBI controls. The hypothesis was also chosen after seeing TEST sign patterns in a campaign with no untouched data. Finally, the L-versus-M pattern in B98 suggests a static anchor difference may already capture most of what the learned version would.
