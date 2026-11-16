# B.32 — two sources of a large fast-channel excursion: the heating beam and an edge burst

Registered 2026-11-16, before the readouts of sections 3 to 5 were computed. CPU only; no model is trained.

## 1. Why

The trained estimator responds to the departure of a fast channel from its mean over the preceding ten rows: linearly up to about two standard deviations and not at all beyond three (manuscript, "What the fast channels are responding to"; the 20-parameter filter learns a threshold of 2.9 to 3.0 standard deviations and a summed BES gain of +48 to +53 eV per standard deviation). Why an estimator should stop following a fast channel that departs far has two physical readings, and the author asked for both to be stated in the paper and checked.

**Beam.** BES is light emitted by atoms of the heating beam, so its level follows the beam within much less than a frame. The ion temperature follows an energy balance: if the heating is removed for a time Δt, the temperature falls by at most the fraction 1 − exp(−Δt/τ), τ being the local ion energy confinement time, and it relaxes over a few τ after the beam returns. (Collisional equilibration between electrons and ions scales with m_i/m_e and is slower still; it does not set this response. Transport does.) In this reading a large, fast change of BES is not accompanied by a change of T_i of proportional size: an estimator that read the BES departure in proportion would over-read it.

**Edge burst.** A burst at the plasma edge raises the edge light within less than a millisecond. A 10 ms file holds one raw sample per row, so a row whose sample falls on a burst departs upward by several standard deviations, while the temperature at the pedestal top changes in those frames for a plasma reason.

## 2. What had been seen before this registration

- The response of T_i, the ECEI level and the BES level to a beam blip (`docs/paper/file_rules.md`): first row after a blip T_i 0.819, ECEI 0.926, BES 0.656 of the level before; T_i and ECEI return over about 90 ms, BES by the fourth row.
- `docs/paper/fast_departure.md` (computed today, all 641 files): in the first row after a blip the BES channels lie a median 2.32 standard deviations below their 100 ms mean and 0.357 [0.319, 0.396] of the channel values lie beyond three; in all other rows 0.004 [0.003, 0.004]. In rows that are not after a blip the mean T_i change by bin of the mean BES departure is +36.7 eV at +0.5 to +1, +42.4 at +1 to +2, +32.9 at +2 to +3; −45.9 at −2 to −1, −109.4 at −3 to −2 and −54.0 [−110.1, +5.0] below −3.
- The MAST coupling study: T_i increments are associated with increments of the power of the viewed beam about 50 ms earlier, and the association sits at beam switching.

Not seen: T_i in the first row after a blip as a function of the BES departure of that row; where the departures beyond three standard deviations outside blip rows sit relative to the bursts of the raw stream; T_i on MAST around a step of the beam that the spectrometer does not view.

## 3. KSTAR, all files: after a blip, does the T_i change scale with the BES departure?

Rows: the first row after an absent frame (a step of 20 to 40 ms inside a segment) with six contiguous 10 ms rows before it, T_i observed in that row and in at least three of the six rows before, values above 3 keV left out. For each such row: the mean over the nine BES channels of the departure d (channel standardized per discharge, minus its mean over the preceding ten rows), and the ratio of T_i to the median of the observed T_i of the six rows before. Groups: D1, d ≤ −3; D2, −3 < d ≤ −2; D3, d > −2. Intervals: 2,000 resamples of discharges.

- **A1.** The median ratio of D1 is not below that of D3: the interval of (D1 − D3) includes zero or lies above it. It fails if the interval lies below zero and the point difference is below −0.05.
- **A2.** In D1 the measured mean T_i change (eV, from the median of the six rows before) is above the proportional reading, 50 eV times the mean d of the group, by at least 30 eV, with the interval of the difference excluding zero.

Expectation: ratios 0.78 to 0.86 in each group, difference within ±0.04; measured change in D1 −90 to −150 eV against a proportional reading of −180 to −260 eV.

## 4. KSTAR, twelve discharges with a raw stream: where do the departures beyond three standard deviations sit?

Rows inside the raw BES window of a discharge and not within ten rows after an absent frame. Bursts: the B.31 detector, unchanged (`b31_bursts/run_b31.py::bursts_and_blips`). U: rows whose mean BES departure is above +3.

- **B1.** The share of U rows with a detected burst within 1 ms of the time stamp is at least three times the share over all rows of the class, and the interval of the ratio (resampling the discharges) excludes 1. Not testable with fewer than 20 U rows.
- **B2 (reported, decides nothing).** For rows with a mean departure below −3: the share whose raw BES level in the 1 ms around the stamp is below 0.8 of its 200 ms running median.

Expectation: 20 to 150 U rows; share with a burst 0.4 to 0.9 against a base share of 0.02 to 0.15.

## 5. MAST: T_i after a step of the beam that the spectrometer does not view

MAST has two beams; the charge-exchange system views the SS beam and beam power is public for both. A switch of the SW beam while the SS beam stays on is a step of the heating that leaves the measurement beam untouched.

Frames: the 5 ms frames of the scored block of a shot (`mast/run_gate.py::prepare`, population G, relative T_i error below 0.1). Beam powers averaged over each frame. Step down at the boundary before frame n: mean SW power of frames n−4 to n−1 at least 0.3 MW, mean of frames n+1 to n+16 at most 0.2 of that. Step up: mean SW power of frames n−4 to n−1 at most 0.05 MW and mean of frames n+1 to n+16 at least 0.3 MW. In both, the SS power of every frame from n−4 to n+16 is at least 0.3 MW and within 20 % of its mean over n−4 to n−1. A chord enters an event if it is good in at least three of the four frames before; its ratio at frame k is T_i(k) over the median of its good values in the four frames before. One value per event and frame: the median over its chords. Shots: every shot of the edge-study list with a block and regular frame times. Intervals: 2,000 resamples of shots.

With R_1 the median ratio at the first frame after the step (k = n+1) and R_late the mean of the median ratios of frames n+10 to n+16 (50 to 80 ms after):

- **C1 (direction).** After a step down R_late is below 1 and after a step up above 1, both intervals excluding 1.
- **C2 (delay).** For each kind of step, |1 − R_late| − 2 |1 − R_1| is above zero with the interval excluding zero: less than half of the late response is present in the first frame.
- **C3 (reported).** The first frame at which |1 − R(k)| reaches half of |1 − R_late|.

A kind of step with fewer than 30 events is not testable. Core (normalized radius below 0.5) and edge (0.7 and above) are reported beside the pooled readout.

Expectation: 50 to 600 events of each kind; R_late 0.90 to 0.98 after a step down and 1.02 to 1.12 after a step up; R_1 within 0.02 of 1; half response at 3 to 8 frames (15 to 40 ms).

## 6. What goes into the paper

- A1 and A2 hold: the paper says that after a blip the T_i change does not scale with the BES departure, that a proportional reading over-reads it, and that not following a departure beyond three standard deviations is the form the beam reading calls for. A1 fails: the paper says that T_i after a blip scales with the BES departure and does not attribute the switch-off to the beam.
- B1 holds: upward departures beyond three standard deviations are described as samples that fall on edge bursts. Fails or not testable: stated as such.
- C1 and C2 hold: the paper says that on MAST, where the beam power is public, T_i follows a step of the heating with a delay of tens of milliseconds. Otherwise the readout is reported as it is.
- Nothing from MAST is written as support for a KSTAR number. The bursts are edge bursts consistent with edge-localized modes, not shown to be such. The mechanism behind the fast-channel increment stays "not identified" unless a statement here says otherwise; these readouts check the two readings, they do not choose between them for every row.

## 7. Limits known in advance

The files carry no beam power; a blip is read from an absent frame. In the first row after a blip the beam is still returning, so the BES departure of that row mixes the beam and the plasma. On MAST a switch of one beam changes fuelling and torque as well as heating, and beam switches are often part of a programmed change of the discharge. The 12 discharges with a raw stream were chosen for other reasons and half of them have few bursts.
