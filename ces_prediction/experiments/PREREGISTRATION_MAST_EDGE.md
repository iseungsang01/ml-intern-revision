# PREREGISTRATION MAST edge — confirmatory test on the unread MAST discharges: does the CES measurement weaken and the T_i regime number fall toward the plasma edge

Registered 2026-11-15, before any value of any shot in the confirmatory set is read. Committed
before the analysis code.

## 1. Why, and where the hypothesis comes from

The KSTAR chord (R = 2.198 m) sits on the edge side: nominal geometry gives a normalized radius of
0.80 (R0 = 1.8 m, a = 0.5 m), the working estimate with a shifted axis is about 0.85, and for one
of our discharges (31923) the literature places a BES channel 4 mm from the chord at r/a = 0.95.
The motivation of the study is that CES is weak there (low signal, failed fits, gaps). KSTAR has one
chord, so the statement "the edge is where the measurement weakens and the increments become
noise-dominated" cannot be tested on KSTAR itself.

The hypothesis below was **found post hoc** in the 300-shot MAST gate
(`docs/paper/mast_gate_addendum2.md`, stratum f; 112-122 TEST discharges): the fit-failure share
is flat inside rho = 0.7 (0.23 / 0.23 / 0.26) and rises outside (0.43 at 0.7-0.9, 0.65 at 0.9-1.0);
the T_i regime number is +0.27 to +0.35 inside 0.7 and +0.103 [-0.079, +0.257] (population G) /
+0.136 [+0.036, +0.230] (W) at 0.7-0.9, -0.112 [-0.296, -0.000] (W) at 0.9-1.0. The registered
monotone statements of that addendum (S6-S8) did not hold as worded; "falls outside 0.7" was read
after the fact. This registration tests that reading on discharges that played no part in it.

## 2. Data

- Confirmatory set: every shot of `mast/all_shots.txt` (the level-1 `act` shots that are neither
  one of the 300 gate shots nor one of the 13 pilot shots) that has `data/mast/act/{shot}.npz`.
  The download (`mast/fetch_mast.py`) must have finished (`data/mast/fetch.log` ends with a line
  starting `done`) before any file is opened.
- State of knowledge: no value of any of these shots has been read by anyone in this project. Only
  file existence and byte counts are known. The catalogue (`l1_shots.parquet`,
  `l1_sources.parquet`) has been read.
- Primary set: catalogue quality "Not Checked". Shots flagged "Bad" are analysed separately and
  co-reported (the gate list excluded them by construction, so nothing is known about them).
- Campaign of each shot from the catalogue (M6-M9), co-reported per campaign.

## 3. Definitions (identical to the gate unless stated)

- Fit rule, populations G (relative T_i error < 0.1) and W (< 0.5), beam block, scored chords
  (good in >= 90 % of the block), usable shot (median frame spacing 5.0 +- 0.1 ms), alternate-frame
  decimation, scored rows, nominal 5 ms frame grid: `PREREGISTRATION_MAST_GATE.md` section 2 and
  the deviations recorded in `docs/paper/mast_gate.md` section 7, as implemented in
  `mast/run_gate.py` (`good_masks`, `prepare`, `regime_number`, the arm functions), reused as code.
- Hyperparameters: the values selected on the gate TUNE shots in addendum 1
  (`docs/paper/mast_gate_addendum1.json`, `retuned`). No retuning. Arms evaluated: persistence,
  running mean (tuned tau = 2.5 ms), slope-state filter, causal GP; plus two fixed running means
  with tau = 15 ms and tau = 40 ms (not tuned; they exist to ask whether averaging helps, since the
  tuned tau is the persistence limit).
- Normalized radius: rho = (R_chord - R_axis) / (R_out - R_axis) with the catalogue EFIT scalars
  `radii_magnetic__radius_efit`, `radii_router_radius_efit` (fallback for R_out:
  `radii_router_radius_d_alpha`), one pair per discharge; low-field side only (0 < rho < 1). Shots
  without geometry, and shots whose radius vector does not match the chord columns, are counted and
  excluded.
- Bins (fixed): 0-0.3, 0.3-0.5, 0.5-0.7, 0.7-0.8, 0.8-0.9, 0.9-0.95, 0.95-1.0, and the merged
  bin 0.9-1.0. Reference bin: 0.5-0.7. Core: 0-0.7 pooled. KSTAR-matched candidates, both
  registered: 0.8-0.9 (nominal geometry 0.80, working estimate 0.85) and 0.9-1.0 (literature
  r/a = 0.95 for a channel 4 mm from the chord).
- Pre-rule samples (for the measurement-quality read-outs): beam block defined on population W,
  all finite T_i samples of the low-field chords inside it, before any fit rule.
  - Fail share: finite samples that fail the rule at relative error 0.1 (and at 0.5), pooled as a
    ratio of sums over shots.
  - Relative T_i error and V_i error: per-shot median in the bin, then the median over shots.
- Regime number per shot and bin: `regime_number` on the scored chords of that bin (median over
  chords of the lag-1 autocorrelation of 10 ms increments), then the median over shots.
- Skill against persistence per bin: 1 - SSE_arm / SSE_persistence on the scored rows of the bin,
  pooled over shots.
- Intervals: 2,000 bootstrap resamples of discharges, seed 20261116, the same resample for every
  bin so that differences between bins are paired. A bin with fewer than 8 contributing discharges
  gets a point value and no interval ("too few").

## 4. Statements

Primary (two):

- **E1 (measurement weakens).** The fail share at relative error 0.1 is larger in each of the bins
  0.8-0.9, 0.9-0.95, 0.95-1.0 than in 0.5-0.7: the three difference intervals lie above 0.
- **E2 (regime number falls).** The T_i regime number, population W (chosen because population G
  removes the weak chords the statement is about; G co-reported), is smaller in 0.8-0.9 and in
  0.9-1.0 than in 0.5-0.7: both difference intervals lie below 0.
  - E2b: in 0.9-1.0 the T_i regime number interval (W) includes 0 or lies below 0.

Secondary:

- **E3.** The causal-GP skill against persistence (T_i, population G; W co-reported) is smaller in
  0.8-0.9 and in 0.9-1.0 than in the core 0-0.7: difference intervals below 0.
- **E4.** Averaging helps toward the edge: the skill of the fixed running mean (tau = 15 ms)
  against persistence (T_i, population W; G and tau = 40 ms co-reported) is larger in 0.9-1.0 than
  in the core (difference interval above 0), and in 0.9-1.0 its interval is not entirely below 0.
  This is the KSTAR direction (a running mean has positive skill over persistence there).
- **E5 (rotation).** Predicted direction: the same as T_i. The V_i regime number (population W) is
  smaller in 0.8-0.9 and in 0.9-1.0 than in 0.5-0.7 (difference intervals below 0). The V_i fail
  share is the T_i one by construction and is not a separate statement.

Co-reported without a statement: every read-out per campaign M6-M9; the "Bad"-quality set; the
causal-GP mean reduced chi-square per bin; the slope-filter skill per bin; counts.

## 5. Numerical expectations (from the gate; stated so they can fail)

- Usable shots with geometry and a W block: 2,000 to 3,000 in the primary set.
- Fail share (0.1): 0.5-0.7 0.22-0.30; 0.8-0.9 0.38-0.52; 0.9-0.95 0.50-0.68; 0.95-1.0 0.62-0.85.
- T_i regime number, W: 0.5-0.7 +0.25 to +0.40; 0.8-0.9 +0.02 to +0.22; 0.9-1.0 -0.25 to +0.05.
  E2 and E2b hold.
- T_i regime number, G: 0.8-0.9 +0.0 to +0.25; 0.9-1.0 not predictable (few chords pass).
- E3: holds for 0.8-0.9 (causal-GP skill there between -0.15 and +0.08, core +0.08 to +0.16).
- E4: fixed running mean (15 ms) in the core -0.45 to -0.20; in 0.9-1.0 -0.15 to +0.10; the
  difference is above 0; whether the edge interval reaches 0 is a coin flip (stated prediction: it
  includes 0).
- E5: V_i regime number W: 0.5-0.7 +0.35 to +0.50; 0.8-0.9 +0.05 to +0.30; 0.9-1.0 -0.05 to +0.25.

## 6. What is written for each outcome

- E1 and E2 hold: "On N MAST discharges not used to form the hypothesis, the share of failed fits
  rises and the T_i regime number falls outside rho = 0.7; at the normalized radius of the KSTAR
  chord MAST is between its own core and KSTAR." The two-device contrast is then written as a
  radius (signal-strength) dependence.
- E1 holds, E2 does not: "The measurement weakens toward the edge, but the fall of the regime
  number read post hoc in the gate was not confirmed"; the device contrast stays as written and the
  gate reading is reported as not confirmed.
- E1 does not hold: the post hoc reading of the gate was not confirmed; stated so.
- E2b alone decides whether the sentence "at the edge MAST reaches the noise-dominated side" may be
  written.
- E3-E5 are reported with their intervals in either direction and do not change the wording above.

## 7. Limits (stated in advance)

- rho uses one EFIT scalar pair per discharge; the axis and the boundary move during the beam
  block. rho is a geometric midplane ratio, not a flux coordinate; r/a in the literature is flux
  based, so "0.95" there and rho here are not the same number.
- The catalogue has no confinement-mode label (the columns `scenario`, `shot_scenario`,
  `shot_experiment_tags` are programme codes), so no discharge is known to have a pedestal and no
  pedestal position is used. The statements are about normalized radius, not about the pedestal.
- 5 ms frames do not resolve an ELM cycle; nothing here is ELM-resolved.
- The fit rule shapes the scored chords: population G keeps few edge chords, which is why E2 is
  read on W. The regime number at the edge is therefore that of the chords that still pass.
- Reported errors are taken as stored; their calibration differs between campaigns (addendum 2).
- Seven statements; E1 and E2 are primary, the rest secondary, no multiplicity correction.
- KSTAR's normalized radius is not EFIT-mapped (0.80 nominal, 0.85 working estimate, 0.95 from one
  published figure for a neighbouring channel).
- CPU only, no network is trained. Output: `docs/paper/mast_edge.json` / `.md`, record in
  `THESIS_RESULTS.md` section 8.
