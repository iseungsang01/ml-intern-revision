# PREREGISTRATION MAST gate, addendum 2 — breakdown by campaign and machine configuration

Registered 2026-11-15, before any stratified value is computed. This is a **post hoc breakdown**:
the pooled TEST results of the gate (`PREREGISTRATION_MAST_GATE.md`, `docs/paper/mast_gate.json`)
and of addendum 1 (wider grids, `docs/paper/mast_gate_addendum1.json`) have been seen. What has not
been seen is any of those readouts split by campaign or by configuration; only the catalogue counts
of the configuration tags over the 300 gate shots are known (heating: 2 beams 82, SS beam 49,
SW beam 17, Ohmic 2, no record 150; RMP coil: true 36, false 72, no record 192; shape: connected
double null 65, lower single null 43, no record 192; divertor "Conventional" 300).

## 1. Why

The two-device contrast (KSTAR noise-limited, MAST trend-limited) was read on a sample that mixes
four campaigns (M6-M9, 2006-2013), beam configurations, RMP on and off, and at least two layouts of
the CXRS arrays (96 and other column counts; three shots with 10 ms frames). A referee can ask
whether the MAST numbers are a property of the plasma or of one measurement set-up. The breakdown
says how far "MAST" can be written without a qualifier.

## 2. What is fixed

Nothing of the gate is changed: shot list, the TUNE / TEST split, populations, scored rows, the
nominal 5 ms grid, arms, the seed 20261116. Hyperparameters are those selected in addendum 1 (wider
grids); there is **no retuning** per stratum. The pooled TEST numbers of addendum 1 must be
reproduced exactly by the new script before any stratum is read (integrity check).

## 3. Strata

- (a) campaign: M6, M7, M8, M9 (catalogue `campaign`). Primary.
- (b) heating (catalogue `heating`): "2 Beams,SS Beam,SW Beam" / "SS Beam" / "SW Beam" / "Ohmic" /
  no record.
- (c) RMP coil (catalogue `rmp_coil`): true / false / no record.
- (d) CXRS layout from the stored arrays: number of chord columns of `act_ss_temperature`; median
  frame spacing (5 ms / other).
- (e) auxiliary, many missing records: `plasma_shape`, `current_range`.

## 4. Readouts per stratum

TEST shots, population G (relative T_i error < 0.1), for T_i and V_i:

- regime number: median over shots of the per-shot lag-1 autocorrelation of 10 ms increments;
- mean reduced chi-square of the best causal arm (the causal GP, as selected pooled in addendum 1),
  reference 1.5; the slope-state filter is reported next to it;
- MSE skill against persistence of the causal GP, the slope-state filter and the running mean;
- co-reported error ratio (median of the squared second difference over its reported-error
  variance, divided by 0.455), reference 1.

Counts over all 264 gate shots with data (TUNE and TEST together), per stratum: usable, no block,
block but no scored chord, block and chords.

Intervals: 2,000 bootstrap resamples of the shots inside the stratum, seed 20261116. A stratum with
fewer than 8 TEST shots carrying scored rows is reported as point values only, marked "too few".

## 5. Statements fixed in advance

- **S1.** The regime number is positive with an interval above 0 in each of the four campaigns, for
  T_i and for V_i. Prediction: holds for V_i in 4 / 4; for T_i the point is positive in 4 / 4 and
  the interval is above 0 in at least 3 / 4.
- **S2.** The mean reduced chi-square of the best causal arm has an interval above 1.5 in each of
  the four campaigns. Prediction: holds for V_i in 4 / 4; for T_i the point is above 1.5 in 4 / 4
  and the interval in at least 2 / 4 (the pooled lower bound was 1.94, so single campaigns are
  expected to be wide).
- **S3.** The running-mean skill against persistence is at or below 0 (interval not above 0) in each
  campaign, both targets. Prediction: holds 4 / 4 (the tuned time constant is the persistence limit).
- **S4.** Gate shots without a beam block are concentrated in discharges without the SS beam.
  Prediction: among shots with a heating record, the share without a block is at least 0.8 for
  "SW Beam" and "Ohmic" and at most 0.2 for records that contain the SS beam. Read on population G
  and co-reported on population W.

Each prediction can fail; the outcome is reported as stated here.

## 6. How the paper wording is narrowed

- S1 or S2 fails in one campaign: "on MAST" becomes "on MAST campaigns X-Y" for that statement, the
  failing campaign is named with its numbers, and the pooled number stays in the table with the
  per-campaign rows next to it.
- The sign of the regime number differs between campaigns: the device contrast is not written as a
  KSTAR-versus-MAST contrast; it is written as a dependence on the regime number, with the campaign
  that is closer to KSTAR named.
- The error ratio differs between campaigns by more than their intervals: the reduced chi-square
  is quoted per campaign and the pooled value is marked as mixing calibrations.
- A configuration stratum (heating, RMP, layout) differs from the rest by more than its interval:
  stated as an observation with the count of shots and the share of missing records; no causal
  reading (the strata are confounded with campaign and with each other, and half the tags are
  missing).
- Nothing differs: one sentence, "the readouts keep their sign in each of the four campaigns", with
  the table in the appendix.

## 7. Disclosures

- Post hoc; multiple strata; no correction for multiplicity. Only S1-S4 are statements; the rest
  is description.
- Strata are small (about 20 TEST shots with rows per campaign in population G) and confounded.
- Catalogue tags are taken as stored; missing records are their own stratum.
- A hardware timeline of MAST for 2006-2013 is collected from the literature in parallel
  (`docs/paper/review_loop/venue_plan/mast_hardware_timeline.md`); it does not enter any rule here.
- Output: `docs/paper/mast_gate_addendum2.json` / `.md`; record in `THESIS_RESULTS.md` section 8.
  CPU only; only gate-list shots are read.
