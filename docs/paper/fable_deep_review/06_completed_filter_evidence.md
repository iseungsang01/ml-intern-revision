# Source audit: RKN, CRU and KalmanNet information sets and update/mask contracts

**Scope and limits.** Claims are tagged [S] directly read source, [A] paper-author claim, [H] application hypothesis, [U] unknown. Code was supplied as unnumbered blobs at pinned commits, so locators are path plus function and I assert no line numbers. Nothing here was run or fitted.

## Locators (six)

- **L1.** RKN PDF p.5, §2.4 Eqs 4–7 and §2.6.
- **L2.** `rkn_cell/RKNCell.py::RKNCell.forward` (ALRhub/rkn_share @7cb29e1).
- **L3.** `lib/CRUCell.py::RKNCell._update` (boschresearch/Continuous-Recurrent-Units @79723e7).
- **L4.** `lib/CRULayer.py::CRULayer.forward` (same commit).
- **L5.** KalmanNet PDF p.3, §II-B.
- **L6.** `KNet/KalmanNet_nn.py::KalmanNetNN.KNet_step` (KalmanNet/KalmanNet_TSP @828a2cf).

## 1. Information sets and contracts

| | RKN | CRU | KalmanNet (original) |
|---|---|---|---|
| Dynamics/observation model | Learned latent transition; the cell takes no f/h argument [S: L2] | Same, with a Δt-dependent prediction [S: L4] | f and h assumed known or approximated; only the noise statistics are unknown [S: L5]. The step calls the supplied f/h through `step_prior` [S: L6] |
| Supervision | Decoder trained on targets | Same | Labelled ground-truth state sequences [S: L5] |
| Estimate at t uses | Inputs ≤ t (forward recursion) | Inputs ≤ t, plus the gap to the next timestamp for prediction [S: L4] | Past and current observations [S: L5] |
| Mask contract | Default config sets `never_invalid=True`, so `forward` calls the unmasked `_update` and ignores `obs_valid`; masking happens only if the flag is switched off [S: L2] | `_update` always applies `where(obs_valid, …)`, leaving mean and all three covariance vectors at the prior when invalid [S: L3] | None: `KNet_step` takes only `y` and applies the gain every step, with no validity or Δt argument [S: L6] |
| Mask granularity | Whole latent-observation vector per step | Same: one boolean per sequence per step, not per feature [S: L3, L4] | n/a |
| Step order | Update, then predict | Update at i, then predict over `time_points[i+1] − time_points[i]` [S: L4] | Prior, then gain, then posterior [S: L6] |

**Filter versus smoother.** All three supplied code paths are forward-only recursions with no backward pass [S: L2, L4, L6], and KalmanNet states its focus as filtering [S: L5]. None is a smoother. Your two-sided PCHIP+GP and the L5 label are the only two-sided objects in the study and must stay offline.

**The paper/code gap worth registering.** The RKN paper says absent inputs are handled by skipping the update [A: L1]. The official RKN cell does this only when `never_invalid` is disabled [S: L2]. The CRU fork makes it unconditional [S: L3]. Any "RKN handles missing CES" statement must therefore name the configuration.

**Known f/h.** This requirement applies to original KalmanNet only [S: L5, L6]. For this dataset KalmanNet additionally lacks ground-truth state labels, since carbon-impurity reports are not bulk-ion truth [H]. This is a statement about applicability, not about performance.

## 2. Pseudo-observation versus a newly arrived CES sample

- **Covariance effect [S: L1, L3].** Every valid update multiplies the upper covariance by (1 − q) with q in (0, 1). A valid update always reduces the stated uncertainty.
- **Stale anchors [H].** If a held or stale CES anchor is presented with `obs_valid=True` at each 10 ms step, the recursion treats each repeat as independent evidence. Variance collapses and the gain to genuinely new information is suppressed. That is a pseudo-observation.
- **Correct contract [H].** The CES update fires only on the grid step where a new report first becomes available. Otherwise the posterior equals the prior.
- **Whole-step mask [S: L3, L4 → H].** BES/ECEI exist every step and CES does not, so a single `obs_valid` cannot serve both streams. There are two options:
  - **Within the source contract:** zero-encode CES plus a flag inside one encoder. The update is then always valid and the structural guarantee is lost, which is functionally your existing anchor/age/flag concatenation.
  - **Departing from the source:** two sequential updates per step, one per stream.
- **Target leakage [S: L4 → H].** The CRU loop updates before producing the posterior for that step. Since the target F is the current report, F_t must not enter any update before the step-t decode.
- **Source time [S: L4 → U].** The source carries one timestamp per row and no separate measurement-time versus arrival-time notion. Actual acquisition and transfer times are unestablished, so only nominal stamping is possible. Source-age rules remain bookkeeping, not forecast horizons.

## 3. Smallest learned-latent candidate

Registration only, not fitted [H]. A discrete factorized RKN-type cell is sufficient, because the grid is a nominal 10 ms and irregular true timestamps are not established. CRU's matrix-exponential prediction buys nothing demonstrable here.

Per grid step t:

1. **Fast update.** An encoder maps the current BES9/ECEI4 raw snapshot to a latent observation and variance. This is always valid.
2. **Decode.** The nowcast and variance are scored against F_t. The information set is past CES plus current and past fast signals.
3. **CES update.** This runs only if a report newly became an anchor, with observation variance a learned function of the flag. It is masked exactly as in L3.
4. **Predict.**

Use a tiny latent (a few units), a matched trajectory budget, and causal expanding standardization as already in place. Update cost is elementwise [A: L1]. Meeting 10 ms on your hardware is [U] until timed.

Limitations:

- Raw voltages are not calibrated ne/Te, so the fast encoder is a learned regressor, not a physical observation model.
- Gaussian NLL on noisy and possibly MNAR report targets will fold report noise and missingness into σ. There is no immunity.

## 4. Simple source-time-consistent comparator

[H] A scalar local-level (random-walk) Kalman filter on the CES channel under the identical contract:

- Update only at genuine new-report steps, nominally stamped.
- Predict-only otherwise, so variance grows with anchor age.
- Two variances fit on TRAIN.
- Fast diagnostics added through the same learner already used for GP+fast.

This isolates the bookkeeping contract from learned representation. It partly overlaps your causal GP and first-order greybox. Its distinct role is being the exact masked-update twin of the candidate.

## 5. What remains different from the hand-coded observer and anchor concatenation

No novelty is claimed: RKN and CRU are published, and this is a familiar model applied to a new information set.

| | Existing | Candidate |
|---|---|---|
| Hand-coded observer | Fixed or scheduled gain | Gain computed per step from propagated covariance and a learned, input-dependent observation variance [S: L1] |
| Anchor concatenation (LSTM/GRU/TCN) | Network may learn to use age and flag | Structure forbids a state update without a new sample [S: L3], and predictive variance widens between anchors by recursion rather than by learned convention |

Whether either difference matters on these data is [U]. Whether a genuinely new algorithmic element could emerge, for example a source-age-aware update, is neither claimed nor excluded.

## 6. Competing explanations and falsifiers

Both are prospective and exploratory. Registration precedes any fit, there is no untouched campaign, and frozen-fit cluster intervals omit fit and selection uncertainty.

**E1: contract explanation.** Any advantage over concatenation models comes from update bookkeeping (no pseudo-observations, age-growing variance), not from learned latent structure.

- *Falsifier:* in a matched twin, the candidate with the CES mask forced always-valid on hold-filled anchors is not worse on the registered paired cluster difference. Equivalently, the masked local-level comparator fails to recover the candidate's gain.
- A non-significant difference alone supports neither side. Register a margin.

**E2: fast-representation explanation.** The advantage comes from the encoder and latent memory extracting an increment from BES/ECEI beyond CES history.

- *Falsifier:* the candidate's paired gain over the comparator persists when the fast inputs are replaced by your existing permutation or commonality controls, or is matched by NBI-only inputs.
- Persistence under these controls would point to capacity or slow-context effects, not a fast diagnostic increment.
- An observational shift in performance still identifies no cause.

## 7. Unknowns carried forward

- Real acquisition and transfer latency.
- Per-frame fit uncertainty, hence whether the learned σ_obs tracks anything physical.
- Wall-clock inference per step.
- Behaviour of NLL under report-target noise.
- All performance on these data. Access to the sources establishes contracts, not results.
