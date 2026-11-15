# PREREGISTRATION MAST coupling — lagged association of CXRS increments with beam power and D-alpha

Registered 2026-11-16, before any relation between the charge-exchange values and the beam-power
or D-alpha traces is computed for any shot. Committed before the analysis code. Why: the paper has
no physics-facing read-out, and the KSTAR rotation result is a bound on coupling to fast channels
(|r| <= 0.044 after noise correction, lags 0-200 ms). A driver acts on the ions with a delay, so the
association is read as a function of lag, not at lag 0 only.

## 1. Data and state of knowledge

- Shots: the primary set of the edge study (`PREREGISTRATION_MAST_EDGE.md`): catalogue quality
  "Not Checked", 5 ms frames, EFIT geometry, radius columns matching. Rebuilt by the same rule:
  3,686 shots, listed in `mast/coupling_shots.txt` (1,846 odd shot numbers, 1,840 even).
- The charge-exchange values of these shots were read by the edge study. The beam traces stored in
  the same files and the D-alpha traces were never read together with them: for the beam only array
  names, shapes and time bases of 40 files were inspected; for D-alpha only byte counts.
- Beam power: `anb_ss_sum_power` (MW, the beam the charge-exchange system views; present in 3,674
  shots) is the primary beam driver; `anb_tot_sum_power` (3,288 shots) is co-reported. Time base
  `anb_time`, 0.05-0.1 ms sampling.
- D-alpha: level 1 `xim/da_hm10_t` ("mid-plane wide-angle view from port HM10", volts, 50,000
  samples per shot) with `xim/time`, downloaded by `mast/fetch_dalpha.py` for the listed shots
  only. The download runs while this is written; the analysis uses every listed shot whose file
  exists when it starts and reports that number. No shot outside the list is fetched.
- Time-stamp convention of the charge-exchange frames (start, centre or end of the 5 ms exposure):
  not stated in the metadata. A stamp is treated as the centre of its exposure; every lag carries
  an uncertainty of half a frame (2.5 ms), and lags below 5 ms are not resolved.
- Discovery / confirmation: odd shot number = discovery, even = confirmation. Every choice (the
  lag at which a coupling is quoted) is made on the discovery set; every quoted number and every
  verdict is from the confirmation set.

## 2. Definitions (fixed)

- Fit rule, beam block and scored chords: as in the gate registration (population G: relative T_i
  error < 0.1, primary; W: < 0.5, co-reported). A shot whose block has a frame interval outside
  5.0 +- 0.1 ms or a non-finite stamp is excluded from this study and counted (the driver has to be
  placed on the stored times).
- Normalized radius rho = (R - R_axis) / (R_out - R_axis), low-field side, from the catalogue EFIT
  scalars as in the edge study. Bins: 0-0.3 / 0.3-0.5 / 0.5-0.7 / 0.7-0.9 / 0.9-1.0, and "all" =
  0-1.0.
- Target increments: on each scored chord, dy(j) = y(j) - y(j-1) for consecutive good frames j-1, j
  of the block; y is T_i (eV) or V_i (km/s). Centred per chord over its valid increments.
- Driver on the frame grid: x(n) = mean of the driver over [t_a + 5 n - 2.5, t_a + 5 n + 2.5) ms,
  t_a the stored time of the first block frame, n from -22 to L + 21 (L = block length); NaN where
  the window holds no driver sample. Driver increment dx(n) = x(n) - x(n-1), centred over the valid
  increments of the shot.
- Lag l in frames, -20 ... +20 (-100 ... +100 ms in 5 ms steps); positive = the driver comes first:
  the pair is (dy(j), dx(j - l)).
- Statistic: pooled within-shot Pearson correlation r(l) = sum(dy dx) / sqrt(sum(dx^2) sum(dy^2)),
  sums over all valid pairs of the chords of a bin and over shots. Interval: 2,000 shot-cluster
  bootstrap resamples (seed 20261116).
- Noise correction (the KSTAR analysis, with the reported fit errors in place of a difference
  estimator): signal fraction f = 1 - sum(sigma(j)^2 + sigma(j-1)^2) / sum(dy^2), r_corr = r /
  sqrt(f), in every bootstrap replicate. r_corr is quoted only where the interval of f lies above
  0.05; the edge study found the reported T_i errors larger than the frame-to-frame scatter, so
  for T_i it may not be quotable. The raw r is the primary read-out: fit noise in dy lowers |r| and
  cannot create a sign.
- Level read-out (co-reported): the same statistic with y (centred per chord) and x (centred per
  shot) in place of the increments.
- Selection on discovery: l* = the lag in 0 ... +20 with the largest |r(l)| on the discovery set,
  per target, driver, population and bin. Quoted: r(l*) and its interval on the confirmation set.
- Circular-shift null (confirmation set, population G): in each of 500 replicates the driver
  increments of every shot are rotated on their own frame grid by a random offset between 21 and
  N - 21 frames (N the grid length); the null statistic is max over the lag set of |r(l)|. Reported:
  the observed maximum over lags 0 ... +20, over lags -20 ... -1, and their null p-values.
- Beam steps: a frame n is a step if |dx(n)| >= 0.25 x (maximum of x in the shot). Co-reported:
  every beam read-out with the driver increments within +- 2 frames of a step removed.

## 3. Statements (each with a direction; each can fail)

Primary (two): C1 and C3.

- **C1. Rotation follows the beam with a delay.** V_i vs `ss_sum_power`, bin "all", population G:
  l* lies in +5 ... +100 ms and r(l*) on the confirmation set is positive with an interval above
  0. Expected: l* 10-40 ms, r +0.02 to +0.12.
  If it fails: "no lagged association of rotation increments with beam-power increments is
  detected within 100 ms on MAST" (with the interval), a second bound next to the KSTAR one.
- **C2. Ion temperature follows the beam, no later than rotation.** T_i vs `ss_sum_power`, bin
  "all", G: r(l*) on the confirmation set positive with an interval above 0, and l*(T_i) <=
  l*(V_i) on the discovery set and again for the lag of the maximum on the confirmation set.
  Expected: l*(T_i) 5-30 ms.
  If it fails: the part that fails is named (no association, or T_i lags rotation).
- **C3. The D-alpha association sits at the edge and at short lag.** T_i vs D-alpha, G: in at
  least one of the bins 0.7-0.9 and 0.9-1.0, r(l*) on the confirmation set has an interval that
  excludes 0 with l* <= 5 ms, and its |r| exceeds the |r(l*)| of the bin 0-0.3 (interval of the
  difference of absolute values above 0). Expected sign at the edge: negative (a D-alpha burst goes
  with a drop of edge T_i); |r| 0.02-0.10.
  If it fails: "the D-alpha association with T_i increments is not larger at the edge than in the
  core at 5 ms resolution", with the bins and intervals.
- **C4. The lag grows inward.** T_i vs D-alpha: l* of the bin 0-0.3 is larger than l* of the
  outermost bin that has an interval excluding 0, on the discovery set and for the lag of the
  maximum on the confirmation set. Low confidence. If it fails: no inward delay is resolved at
  5 ms frames.
- **C5. Direction check.** For C1 (V_i, beam, "all", G) the maximum of |r| over the negative lags
  (driver later) has a circular-shift p-value above 0.05, while the maximum over the positive lags
  has one below 0.05. If it fails: the association is not one-sided in time and is not read as a
  response to the driver.

Co-reported for every target and driver: the full lag profile per bin on both sets, V_i vs D-alpha,
population W, `tot_sum_power`, the level read-out, the step-removed beam read-out, r_corr where
quotable.

## 4. Strata (confirmation set, read at the discovery l*; not registered tests)

Campaign M6-M9; catalogue `heating` ("SS Beam" vs "2 Beams,SS Beam,SW Beam", others and missing
left out); `rmp_coil` True vs False (missing left out). A stratum with fewer than 8 shots is not
given an interval.

## 5. Limits stated in advance

- Charge exchange exists only while the viewed beam is on, so the beam trace and the existence
  and quality of the measurement are tied: the switch-on precedes the block, the switch-off ends
  it, and a change of beam power changes the charge-exchange signal level and with it the fit. An
  association with the beam is therefore not only a plasma response. The step-removed read-out and
  the direction check C5 are the two controls available here; neither removes the tie.
- Lags below 5 ms are not resolved and the stamp convention is unknown (half a frame).
- The D-alpha view is a mid-plane wide-angle chord, not local to any charge-exchange chord; no
  "co-located" statement is made.
- No confinement-mode label; rho is a geometric ratio from one EFIT scalar pair per discharge.
- Seven statements, two primary, no multiplicity correction beyond the discovery / confirmation
  split and the circular-shift null.
- No MAST number from this study is used as support for the KSTAR fast-diagnostic claim. It is
  placed next to the KSTAR rotation bound as a counterpart on a different device and radius.

## 6. Output

`mast/run_coupling.py` -> `docs/paper/mast_coupling.json` / `.md`; record in `THESIS_RESULTS.md`
section 8. CPU only, one process, shots streamed one at a time.
