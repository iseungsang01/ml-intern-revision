# Adversarial review of PLAN_draft.md (2026-11-15)

Reviewer stance: hostile, evidence first. Read: `PLAN_draft.md`, `nf_bar.md`, `neurips_bar.md`,
`open_data.md`, `assets.md`, `../round_2.md`, `../../short/main_short.tex`. No repository file other
than this one was written; nothing was committed; no model was trained.

## 0. Verdict in five sentences

1. The MAST decimation task is real but it is **not the KSTAR task**. Offline, the median hidden
   frame is already recovered inside the reported fit error; causally, the headroom is trend
   extrapolation of a ramping 0.3 s discharge, which a CES-only smoother with a slope state can
   probably take.
2. **"What co-located fast diagnostics add" is not testable on MAST** for the general population:
   Thomson samples every 4.0-4.2 ms (one sample per CXRS frame, so it is not a fast channel), there
   is no ECE, BES exists in about 29 % of shots without a radius, the rest is edge, global or
   line-integrated.
3. Fork A's claim sentence ("on two devices, what co-located fast diagnostics add ...") therefore
   cannot be supported by its own Phase 1. After this plan the paper is a KSTAR single-chord result
   plus a MAST history-only study: PPCF or FED material, not Nuclear Fusion.
4. The two devices sit in opposite signal-to-noise regimes (measured below). That is a risk for
   "replicate the audit" and at the same time the one cross-device result the plan can deliver
   cheaply.
5. Keep MAST, shrink it to a gated CPU study, move the KSTAR data request to day 0, and aim at
   PPCF; treat NF as conditional on KSTAR multi-chord data and co-authors, nothing else.

## 1. What was measured today on FAIR-MAST

Source: level 1 `act` (level 2 `charge_exchange` carries no error arrays), anonymous HTTPS,
`s3.echo.stfc.ac.uk/mast/level1/shots/{shot}.zarr`. 13 shots: 12 drawn at random (3 per campaign
M6-M9, seed 20261115, from the 4,031 "Not Checked" shots that also have the level-2 group) plus
30147. Transfer 24.2 MB in 4 min, single stream (18 s per shot). Scripts and the pickle are in the
session scratchpad, not in the repository. Shots: 17703, 18330, 18064, 21770, 24411, 22622, 26581,
27244, 27288, 30448, 30013, 28991, 30147. **These 13 are now seen; a registration must exclude them
or label them pilot.**

Definitions. Good fit: finite, T_i > 0, reported error > 0, relative T_i error below a cut (0.1 or
0.5), T_i < 5 keV. Beam block: first to last frame with at least 10 good chords. Scored chords:
good in at least 90 % of the block. Hidden frame t; visible t-5 ms and t-15 ms (causal) and t+5 ms
(offline). Reduced chi-square = squared residual divided by the variance that the reported errors
alone predict for that estimator; a noise-only Gaussian residual gives mean 1.00 and median 0.455.
Intervals are shot-clustered bootstrap (2,000 resamples) over 9 or 12 shots, so they are wide.

### 1.1 Frames, missingness, beam, Thomson

| Quantity | Value |
|---|---|
| Frame spacing | 5.00 ms in 13/13 shots |
| Frames with at least 10 good chords, per shot | 0, 36, 37, 40, 49, 52, 54, 57, 60, 64, 65, 76, 86; **median 54** (0.27 s) |
| Visible history after decimation to 10 ms | median 27 frames per shot; 27 scored frames per chord |
| KSTAR for comparison (150 random files) | median 363 labelled T_i rows per file (10th-90th percentile 112-668) |
| Shots with no usable frame | 1 of 13 (28991: 98.9 % of finite values fail the fit rule) |
| Failed-fit share of finite values on illuminated chords (relative error >= 0.5 or T_i >= 5 keV), per shot | 2.1 % to 24.7 %, median 7 % |
| Temporal NaN share on the 61-64 illuminated chords inside the beam block | 0.2 % to 9.9 %, median 1.2 % |
| NaN share over all 96 or 160 stored chord columns inside the block | 34 % to 64 % (unused columns, not gaps in time) |
| Good frames while the SS beam is below 0.3 MW | 0 or 1 frame in 10 of the 11 shots with a readable beam trace; in shot 17703 the beam trace disagrees with the catalogue beam times (not resolved) |
| Core Thomson cadence (`ayc/time`) | median spacing 4.00-4.17 ms in 8/8 shots that carry it (240 Hz) |
| BES in level 1 | absent in 13/13 (level 2 only, as `open_data.md` says) |

Reading: CXRS exists only in the beam window, and inside it there is almost no natural missingness
in time (1.2 % against KSTAR's 8.2 % for T_i and 23.9 % for V_rot). The KSTAR framing "fill rows
where CES delivered nothing" has no natural counterpart on MAST; every gap there is one we make.

### 1.2 Is one hidden 5 ms frame trivial? T_i

| T_i, pooled hidden frames | relative error < 0.1 (9 shots, 15,235 rows) | relative error < 0.5 (12 shots, 27,049 rows) |
|---|---|---|
| Median T_i / median reported error | 950 eV / 25.2 eV | 872 eV / 25.4 eV |
| Lag-1 autocorrelation of 5 ms increments, median over shots (range) | **+0.16** (-0.17 to +0.42) | -0.01 (-0.14 to +0.46) |
| Lag-1 autocorrelation of 10 ms increments, median over shots | **+0.33** | +0.07 |
| Offline linear interpolation: median absolute residual / RMSE | 17.3 eV / 61.1 eV | 17.1 eV / 83.2 eV |
| Offline interpolation, reduced chi-square: median / mean | **0.31** / 2.34 [1.68, 3.22] | 0.24 / 2.20 [1.30, 3.53] |
| Causal persistence (5 ms old): median absolute residual / RMSE | 41.2 eV / 84.1 eV | 38.2 eV / 112.2 eV |
| Causal persistence, reduced chi-square: median / mean | 1.34 / 6.09 [4.72, 7.81] | 0.84 / 5.46 [3.65, 7.94] |
| Causal linear extrapolation: median absolute residual; MSE skill vs persistence | 30.3 eV; -0.365 | 29.3 eV; -0.521 |
| Causal mean of the last two visible frames: MSE skill vs persistence | **-0.687** | -0.383 |
| Offline interpolation, MSE skill vs causal persistence | +0.473 [+0.302, +0.601] | +0.450 [+0.355, +0.573] |
| Ceiling on any estimator's MSE skill vs persistence, if the reported errors are right | +0.779 [+0.712, +0.853] | +0.454 [+0.052, +0.757] |

Frame-to-frame scatter against reported error: the RMS 5 ms difference is sqrt(6.09) = 2.5 times
(good fits) and sqrt(5.46) = 2.3 times (all fits) what the errors alone predict; at the median row
the factor is sqrt(1.34 / 0.455) = 1.7 and sqrt(0.84 / 0.455) = 1.4.

KSTAR on the same statistics (136 files, 49,793 rows, T_i <= 3 keV): lag-1 autocorrelation of 10 ms
increments **-0.41** (white noise on a flat signal gives -0.5); interpolation vs persistence MSE
skill +0.266; median absolute residual 34.7 eV (interpolation) against 41.8 eV (persistence).

What this says:

- **Offline, the median frame is at or below the noise floor.** Median reduced chi-square 0.31 is
  under the noise-only value 0.455. Either the reported errors are too large, or adjacent frames
  share noise (a common background frame, temporal smoothing in the fit). The second reading would
  mean the hidden frame is not an independent measurement, which attacks the plan's selling point
  "a measured answer at the filled instant". Not resolved today; it must be resolved before any
  registration (cross-frame and cross-chord residual correlation; a question to the maintainers).
- **The mean is carried by a tail.** Mean 2.34 against median 0.31, a ratio of 7.5 where a Gaussian
  gives 2.2. The tail-vs-median part of the audit will probably be seen on MAST as well.
- **Causally there is headroom, and it is a trend.** Increments are positively autocorrelated, a
  slope term lowers the median error from 41 to 30 eV, and averaging (the cheap move that takes
  about half the PCHIP margin on KSTAR) is the worst arm at -0.69. MAST discharges last about
  0.3 s and are not stationary. KSTAR is the opposite regime: noise-dominated increments, where
  averaging helps. "Largely recovered by 1-4-parameter causal smoothers" can hold on MAST only if
  the smoother has a slope state, and "a running mean takes half the margin" should be expected not
  to hold.
- **Population sensitivity is large.** Admitting fits with relative error up to 0.5 moves the
  ceiling from +0.78 to +0.45 with a lower bound of +0.05. The fit-quality rule decides the result,
  the same problem round 2 found with the 1,000 km/s rule. On MAST the rule can at least be an
  instrument criterion (reported error, `ss_chisq`) fixed in advance.

### 1.3 Rotation on MAST is not the KSTAR rotation problem

V_i, relative-error cut 0.1 (9 shots, 15,235 rows): median 63.7 km/s, median reported error
0.96 km/s; lag-1 autocorrelation of 5 ms increments +0.40 (median over shots; 12-shot set +0.34);
offline interpolation median absolute residual 1.31 km/s, reduced chi-square median 1.11 and mean
19.6 [7.9, 42.1]; persistence median absolute residual 3.65 km/s, reduced chi-square median 6.05
and mean 48.9 [28.2, 79.5]; interpolation vs persistence MSE skill +0.669 [+0.567, +0.737].

Frame-to-frame rotation changes are several times the reported error (factor 3.6 at the median
row). KSTAR Claim 2 (rotation from history only, at a fit-noise floor, coupling |r| <= 0.044) has no
reason to transfer. On MAST the rotation signal is well above its floor, so a coupling to fast
global channels (Mirnov, soft X-ray) is measurable in principle. The draft does not use this
(section 4).

## 2. Objections

**O1. Fork A's claim is untestable on the second device.** Evidence: section 1.1 (Thomson at
240 Hz, one sample per CXRS frame), `open_data.md` 2.5-2.7 (no ECE; BES in 23/80 shots with no
radial position; D-alpha, Mirnov and the interferometer are edge, global or line-integrated; soft
X-ray is line-integrated and costs +120 GB). MAST can support: CES-history-only estimation, the
audit, per-radius noise structure, and global-channel coupling. It cannot support "co-located".
The draft's Phase 0 kill rule says the physics route loses its main axis if the KSTAR fast-channel
increment includes 0, yet Fork A stays primary; since MAST cannot carry that axis either, Fork A
would then have no axis on either device.

**O2. "Replication" is the wrong word.** KSTAR: 10 ms, natural gaps, noise-dominated increments
(-0.41), conventional tokamak, one chord, flat tops of seconds. MAST: 5 ms, manufactured gaps,
trend-dominated increments (+0.16 at 5 ms, +0.33 at 10 ms), spherical tokamak, 64 chords, 0.27 s of
beam. A referee will read two different problems under one model name. Written as a contrast of
regimes it is defensible; written as replication it is not.

**O3. The scored frame may not be independent of its neighbours** (median reduced chi-square 0.31,
section 1.2). Unresolved, and it decides whether decimation gives ground truth at all.

**O4. Per-discharge standardization breaks.** `CES_PER_SHOT_NORM=1` uses whole-discharge
statistics. On a 54-frame ramping discharge that leaks the future (the mean of a ramp says where the
ramp ends). The causal variant (assets item 1b) is a precondition for MAST, not an option. With 27
visible frames a 3-5 frame warm-up is 11-19 % of the scored rows of every shot. History length
itself is not the blocker: the KSTAR backbone's effective memory is about 1.3 steps (B.28 report).

**O5. Event-conditional scoring is thinner than it looks.** A 5 ms frame cannot resolve an ELM
cycle (the limit `nf_bar.md` section 3 states for KSTAR at 10 ms applies here too; MAST ELM
frequencies were not checked today). L-H transitions give at most about one event per shot. A
D-alpha event detector on a device never handled, validated against nothing, is weeks of work.

**O6. Nuclear Fusion remains out of reach on this plan.** `nf_bar.md` lists five necessary items:
a consumer or physics application, more than one KSTAR chord, a second KSTAR campaign, a deployable
pipeline, KSTAR authorship. The plan closes one (deployable, Phase 0) and offers an item that
`nf_bar.md` files under "nice to have" (second device) in place of three necessary ones. No surveyed
NF paper was carried by a margin over an interpolator; a second margin on a second device does not
change the category. "Skill vs radius" is a property of the estimator and the fit noise, not of the
plasma. A transport physicist would care about an ion heat pulse, a transition, or a
momentum-braking event read from the series, and the plan produces none. Estimate after the full
plan: NF Letter below 10 %; PPCF plausible if one physics reading lands; FED as floor.

**O7. Timeline.** "Converter 200-300 lines, 2-3 weeks" leaves out: level-1 errors in addition to
level 2 (18 s per shot single stream measured, about 21 h for 4,278 shots, so parallel transfer and
retries); 96 against 160 chord columns between shots, so a per-shot radius map; r/a from the level-2
equilibrium; a registered fit rule; the independence check of O3; causal standardization; an event
detector. The GPU is not the constraint (27-step sequences). CPU is: the per-row causal GP that
dominates KSTAR scoring (2.6-6 min for about 250 k rows) grows to tens of hours at 4,278 shots x 54
frames x 40-60 chords unless chords are subsampled. Estimate for items 4-7: 5-8 weeks for one
person. Fork B's 14-18 weeks on top does not fit before May 2027 alongside Fork A.

**O8. The KSTAR request is on the critical path but is scheduled "in parallel".** Its lead time is
outside the student's control and every NF-necessary item depends on it.

## 3. Verdict per plan element

| Element | Verdict | Reason |
|---|---|---|
| Phase 0 item 1 (history-only retrain) | **keep, first** | Central attribution; decides the title |
| Phase 0 item 2 (`CES_PER_SHOT_NORM=0`) | **change** | Add the expanding-window standardizer (assets 1b); required for MAST (O4) |
| Phase 0 item 3 (CPU checks, memoryless baselines) | keep | Cheap; asked for in both rounds |
| Phase 0 kill rule | **change** | State what Fork A becomes when the increment includes 0; today it stays primary by default |
| Pivot to FAIR-MAST | **change** | Keep as a contrast-of-regimes and audit arm; drop the "second device for the fast-channel claim" role |
| Phase 1 item 4 (converter, 21 GB slim set) | **change** | Start with level-1 `act` values and errors plus beam power for 300 shots (about 0.6 GB); full set only after the gate |
| Phase 1 item 5 (decimation with seq_v2 and fast diagnostics) | **change** | CES-only arms first, including a slope-state Kalman filter; a neural arm only if those leave the reduced chi-square above 1 |
| Phase 1 item 6 (skill vs radius; events) | **change** | Radius: keep (cheap; per-chord errors exist). ELM-conditional: drop. L-H: keep only if a detector validates on hand-checked shots |
| Phase 1 item 7 (audit replication) | keep | Tail-vs-median is likely to show (ratio 7.5); the running-mean statement is not; register both directions |
| "Registered before scoring" | **change** | Exclude the 13 pilot shots; fix the fit rule from `act` errors and `ss_chisq` beforehand |
| Fork A claim sentence | **drop as written** | Not testable on MAST (O1) |
| Fork A venue ladder | **change** | PPCF primary; NF only if KSTAR multi-chord data, events and co-authors arrive |
| KSTAR data request "in parallel" | **change** | Day 0, one list, as `nf_bar.md` section 5 words it |
| Fork B (NeurIPS E&D audit) | keep, secondary, gated | MAST adds the one dataset with per-row reported errors, so the floor-normalized readout can be checked directly; odds unchanged (12-18 % if it replicates) |
| "Why A is primary" | keep | The thesis-framing argument stands |

## 4. The one concrete alternative

Replace "replicate seq_v2 with fast diagnostics on MAST" by a **model-free, two-device regime and
coupling study** (CPU only; reuses the A3 code):

1. Regime number per device, chord and discharge: the lag-1 autocorrelation of target increments
   and the ratio of increment variance to reported fit variance. Measured today: KSTAR T_i -0.41;
   MAST T_i -0.01 to +0.16; MAST V_i +0.34 to +0.40. Statement to test: this number predicts which
   causal comparator is the right bar (running mean, persistence, slope filter) and the ceiling on
   any estimator's skill. On MAST the ceiling can be checked against reported errors at 64 radii,
   which KSTAR cannot offer.
2. Noise-corrected coupling of V_i and T_i increments to Mirnov and D-alpha (soft X-ray on a
   subset) at 64 radii, with the A3 analysis unchanged. KSTAR gave |r| <= 0.044 for rotation at a
   fit-noise floor. MAST rotation is several times above its floor, so a nonzero coupling at
   MHD-active radii would be a positive counterpart, and a bound would be a second bound; either
   outcome is reportable as an interval. Rotation braking by MHD is a topic MAST physicists
   recognise (prior MAST work exists; from memory, not searched today).

This is the physics-facing result the two devices can produce without co-located fast channels. It
is still a diagnostics-and-methods result, not a transport result. It fits PPCF, and it serves
unchanged as the MAST section of the audit paper.

## 5. Revised plan (25 lines)

1. Day 0: send the single KSTAR request (all CES chords with fit metadata; D-alpha, NBI, EFIT;
   a 2023/2024 subset; authorship and data statement). Every NF-relevant item waits on it.
2. Days 1-3, KSTAR, GPU: history-only seq_v2 (4 splits x 2 populations); `CES_PER_SHOT_NORM=0`
   and the expanding-window standardizer; round-2 CPU checks; memoryless MLP and GBM.
3. Branch: if the fast-channel increment interval excludes 0, the KSTAR paper keeps its title;
   if it includes 0, retitle to noise-limited estimation of the CES past. Step 4 follows either way.
4. Day 4: write the MAST registration: exclude the 13 pilot shots; fit rule from `act` errors and
   `ss_chisq`; the independence test of hidden frames; arms, readouts, and both directions of the
   running-mean statement.
5. Week 2 (MAST gate, CPU, 300 shots, level-1 `act` + beam power, about 0.6 GB): independence
   test; CES-only arms (persistence, slope extrapolation, running mean, slope-state Kalman, causal
   GP, offline interpolation and GP) with floor-normalized skill per radius; the regime number.
6. Gate: (a) hidden frames are independent of their neighbours; (b) the best CES-only causal arm
   leaves a mean reduced chi-square whose interval lies above 1.5 on good fits. If (a) fails, MAST
   is used for the regime and coupling study only. If (b) fails, no neural arm is trained on MAST.
7. Weeks 3-4: coupling study (section 4, item 2) on the slim set for the gated shot list; audit
   readouts (drop-k, mean vs median, ladder). Neural arm with Mirnov and D-alpha only if 6(b) passed.
8. Week 5: decide the venue from what exists.
   - KSTAR multi-chord data and events arrived and co-authors agreed: NF Letter becomes a candidate.
   - Otherwise: PPCF, built on the regime contrast and the two coupling results.
   - Phase 0 negative and MAST gate failed: FED, with the audit as the contribution.
9. Fork B starts only after step 7, only if the `neurips_bar.md` kill criteria pass on MAST plus
   two PhysioNet sets, and only if the advisor permits release of the KSTAR per-row error arrays.
10. Not done at any point: ELM-resolved claims, "co-located" claims on MAST, the BES download
    (105 GB, no radius), a full 21 GB pull before the gate.

## 6. Limits of this review

- 13 shots; intervals are over 9 or 12 discharges. Every direction stated above should be rechecked
  on the 300-shot gate before it is quoted.
- The reported-error arrays were taken at face value; their calibration is what O3 questions.
- MAST ELM frequency, the MAST rotation-braking literature and the BES radial scan are from memory.
- Shot 17703's beam trace and shot 22622's unreadable beam chunk were not resolved.
- Linear interpolation stands in for PCHIP, and no GP or Kalman arm was run; the "slope-state
  smoother can take the headroom" statement is an inference from the autocorrelation and the
  extrapolation medians, not a measurement.
- Venue odds are judgments, not measurements.
