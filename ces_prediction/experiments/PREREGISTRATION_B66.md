# B.66 — the increment by regime: discharges and phases with and without edge bursts

Registered 2026-11-18, before any readout of this batch. CPU only, stored errors, no model run. Designed by the session lead after the thirteenth independent grading; implemented and run by an AI agent under the author's supervision.

## 1. Why

The 639 discharges mix confinement regimes (L-mode, ELMy H-mode, enhanced modes, RMP suppression) under one label, "pedestal top". The files carry no regime label and no D-alpha. The 10 ms BES level shows edge bursts (the detector of B.31, used again in B.54 and B.58 for quiet rows); a discharge with frequent bursts has an ELMy-like edge, one without bursts is L-mode or a burst-free H-mode. The grading asks whether the gain belongs to one regime.

## 2. Seen before

B.54: quiet rows carry 0.874 [0.734, 1.098] of the reduction against the twin, increment there +0.054 [+0.026, +0.082]. The increment by discharge temperature third: coldest +0.007 [-0.023, +0.031], middle +0.042, hottest +0.092. Not seen: the increment by discharge regime; whether burst frequency and temperature level give the same split.

## 3. Design

Pooled rows of B.39 (r2 to r9, cut population), stored errors of the network, twin and causal GP; KFX and KF0 on r3 to r9. The burst detector of B.31 unchanged (as used by B.54), run on every file, beam blips excluded as in B.54. Per discharge: burst rate = burst onsets per second of rows. Classes, fixed from the rates over all 639 discharges before any error is read: (A) no burst; (B) rate above zero up to the median of the non-zero rates; (C) above that median. Within discharges of classes B and C: rows within 30 ms after a burst onset (burst phase) and rows at least 50 ms from any onset (inter-burst phase).

Readouts per class and phase: increment over the twin, skill over the causal GP, KFX over KF0, with intervals as B.54 (re-partition drawn, discharges resampled; 20,000 draws); numbers of discharges and rows; median T_i of the class. A two-way table class by temperature third (thirds as B.54), with the increment in each cell having at least 30 discharges.

Checks before any statement: the detector reproduces the quiet-row share 0.889 of B.54 on its rows; the class counts are printed before any error is read.

## 4. Statements

- **R1.** Increment over the twin in class A (no bursts): interval above zero.
- **R2.** Increment over the twin in class C (frequent bursts): interval above zero.
- **R3.** Increment over the twin in the inter-burst phase of classes B and C: interval above zero.
- **R4.** The increment differs between class A and class C: interval of the difference excludes zero.

## 5. What goes into the paper

A small table by regime class in the results on where the gain is. R1 and R2 met: the gain is not tied to one regime. Only R2 met: the gain is established in bursting (ELMy-like) discharges and not shown in burst-free ones; only R1: the converse. R3 states whether the gain is between bursts. If the classes mostly reproduce the temperature thirds (two-way table), the paper says so and claims no regime effect separate from the temperature level. Classes are named by burst rate, not by a confinement mode, since no mode label exists.

## 6. Expectations

Class A about a third of the discharges and colder. R1 uncertain (+0.00 to +0.04). R2 met (+0.05 to +0.09). R3 met. R4 uncertain.

## 7. Limits known in advance

Burst rate is a proxy for an ELMy edge, not a mode label. Classes are confounded with temperature level and beam power.
