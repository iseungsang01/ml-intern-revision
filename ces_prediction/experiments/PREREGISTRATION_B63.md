# B.63 — does the equilibrium change at a beam interruption? The integrated Mirnov signal at two toroidal positions on the discharges with raw streams

Registered 2026-11-18, before any readout of this batch. CPU only, no model. Designed by the session lead after a published statement on the beam interruptions at KSTAR was found; run by a delegated agent with the event code of B.58 and B.62.

## 1. Why

After an interruption of the heating beam the reported T_i of the chord is 0.814 of its level before in the first frame and the ECEI level begins to fall while the beam is off and is lowest 10 to 15 ms after its return (B.58, B.62). A loss of a third of the beam heating for 10 ms gives, by published formulas, a thermal response of about 2 % that is largest 26 to 31 ms after the return: too small and too late for the fall of T_i. A paper on an earlier KSTAR campaign states that the beam interruptions made for the charge-exchange measurement "cause a plasma displacement and ion temperature fluctuations" (Kim et al., Phys. Plasmas 25, 122511, 2018; verified on the PDF by the session lead), and in its figure the interrupted source carries about a third of the beam power. A displacement is a reading that the batches so far could not test: the total stored energy, fast ions included, starts to fall at the rate of the missing beam power as soon as the beam is off (about 1 MW for 10 ms against a stored energy of a few hundred kJ, a few per cent), the outward shift of the flux surfaces follows the stored energy, and a chord at a fixed major radius in a gradient then reads a lower value. Such a change is largest at the return of the beam and decays with the confinement time, which is the time course of the T_i fall. The 10 ms files hold no equilibrium, but the raw streams of twelve discharges hold two Mirnov coils at two toroidal positions at 2 MHz. A coil measures the time derivative of the poloidal field at the wall; its time integral follows the field, and a change of the equilibrium (stored energy, position) appears in it alike at both toroidal positions, whereas a rotating mode does not.

## 2. Seen before

B.62: 41 interruptions of one absent frame in the twelve discharges with raw streams, off for 10.5 ms at the median; the ECEI level 0.972, 0.956, 0.938, 0.944 of its level before in the last 3 ms before the return and 0 to 3, 10 to 13, 20 to 23 ms after it. Earlier records: the 10 ms Mirnov columns are single raw samples and carry no usable information for the estimator; raw Mirnov features did not help the rotation estimate; the raw Mirnov stream of one discharge shows a published coherent mode (positive control of the delivery). Not seen: the raw Mirnov signal integrated in time, at any event.

## 3. Design

**Events.** The interruptions of B.62 (times at which the beam goes off, t_off, and returns, t_on, from 0.1 ms bins of the raw BES, code of B.58). Statements use the interruptions of one absent frame; those of two and three absent frames are reported apart. Pseudo-events for the control: times in the same discharges with no interruption within 200 ms, as in B.62, with the median off-time as their length.

**Integrated signal.** For each event and coil c (MC1T03, MC1T16): the raw signal minus its mean over the 30 ms before t_off, summed from t_off onward and multiplied by the sampling interval, Phi_c(t). By construction Phi_c is zero at t_off and has no trend before it. The common mode is C = (Phi_03 + Phi_16)/2, the difference D = (Phi_03 - Phi_16)/2. Each is divided, per discharge, by the median absolute value of C at the pseudo-events of that discharge in the window that corresponds to "0 to 3 ms after the return" (the control scatter of the discharge), so that all quantities are in units of that scatter. Windows (means over the window): the last 3 ms before the return; 0 to 3, 10 to 13, 20 to 23 and 40 to 43 ms after the return. A profile in 1 ms bins from 30 ms before t_off to 80 ms after t_on is reported.

**Sign.** The polarity of the coils is not known. All statements are on a deviation from zero, not on its sign; the sign found is reported.

**Intervals.** Event medians; 2,000 resamples of discharges.

**Checks before any statement.** (a) The event list reproduces B.62 (41 interruptions of one absent frame, median off-time 10.5 ms). (b) Control: at the pseudo-events the event median of C in every window is within 0.5 control scatters of zero. (c) Sensitivity (positive control): at the 200 largest positive excursions of the 1 ms BES common mode of these discharges that lie at least 50 ms from any interruption (candidate edge bursts; the BES common mode is the mean of the nine channels divided by its running median over 100 ms), the event median of the absolute change of C between the 5 ms before and the 5 ms after the excursion is at least twice that at the pseudo-events. If (c) fails, the integrated signal of two coils does not resolve a known change of the equilibrium, and the statements are reported as not readable. (d) The discharge whose raw Mirnov coverage is partial is used where it covers the event; the number of events per discharge after this is reported.

## 4. Statements

- **M1.** Inside the interruption (last 3 ms before the return of the beam) the common mode differs from zero: the interval of the event median of C excludes zero.
- **M2.** The change is alike at the two toroidal positions: in the window 0 to 3 ms after the return the event medians of Phi_03 and Phi_16 (each in control-scatter units) have the same sign and each interval excludes zero.
- **M3.** The change decays: the absolute event median of C is smaller 40 to 43 ms after the return than 0 to 3 ms after it; the interval of the difference (later minus earlier, of the signed C multiplied by the sign of the earlier median) is below zero.
- **M4.** The difference of the two coils shows no such change: the interval of the event median of D in the window 0 to 3 ms after the return includes zero. (A rotating mode would appear in D, a change of the equilibrium in C.)

Reported, not statements: the 1 ms profiles of C and D with the ECEI and BES profiles of B.62 beside them; the time of the largest deviation of C; C for interruptions of two and three absent frames (a dose in the stored energy would make it larger); the rank correlation over events between C at the return and the ECEI fall 10 to 13 ms after it; every quantity per discharge.

## 5. What goes into the paper

- (c) fails: one sentence in the supplement that two Mirnov coils do not resolve it; the displacement stays an untested reading named from the literature.
- M1 and M2 met (with (b) and (c) passed): the paper states that the poloidal field at the wall changes through the interruption alike at two toroidal positions, that is, the equilibrium changes at the interruptions, with the time course found (M3, the time of the largest deviation); beside it the published statement of a displacement at KSTAR beam interruptions and the sentence that a chord at a fixed radius in a gradient then reads a changed value, of a size that cannot be given without an equilibrium reconstruction. The fall of the reported T_i is then described as occurring together with a change of the equilibrium; it is still not used as evidence on ion heat transport, and the first-frame excess of B.62 stays the bound on a measurement-side part.
- M1 or M2 not met with (c) passed: the paper states that no change of the wall field is resolved at the interruptions at the sensitivity shown by the control, and gives that sensitivity in units of the change at the candidate bursts.
- M4 not met: stated; the change is then not purely axisymmetric and the reading as an equilibrium change is weakened accordingly.
- In every case no size of a displacement in millimetres is given.

## 6. Expectations

Check (c): passes (edge bursts change the stored energy by per cents; ratio 3 to 10). M1: met, a deviation of 1 to 4 control scatters; it may fail if the integration drift over 10 ms is larger than the change. M2: met. M3: met (smaller by a half or more at 40 ms). M4: met (interval includes zero). Largest deviation of C within 5 ms of the return. For longer interruptions a larger deviation (two to three times). The session lead's expectations in B.62 were wrong in all three parts of its raw readout; these are held with the same caution.

## 7. Limits known in advance

Two coils of unknown gain, orientation and poloidal position: no calibration, no conversion to a displacement, and two toroidal positions do not exclude every non-axisymmetric change. The integral of a coil signal drifts; the baseline is the 30 ms before the interruption, in which the equilibrium itself evolves and edge bursts occur. Twelve discharges chosen for other purposes, 41 events. The plasma control system reacts to the same change and its coils couple to the Mirnov coils. A change of the field at the wall shows that the equilibrium changes; it does not show how much of the fall at the chord is due to it.
