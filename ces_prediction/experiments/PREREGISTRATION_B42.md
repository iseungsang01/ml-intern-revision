# B.42 — the estimator when the CES value arrives late: a dense but delayed CES history

Registered 2026-11-18, before any of the readouts below was computed. CPU only; the frozen networks are run forward, nothing is trained. Designed by the session lead and executed by a delegated agent from this document.

## 1. Why

Every readout so far assumes that the CES value of the previous frame is available when the present frame starts. The tenth independent grading: the readout of the spectrometer and the fit take time, the delay is taken as zero, and at the natural cadence 97.4 % of the scored rows have a measurement 10 ms old, so the headline is a one-step prediction of a noisy sample. If the CES analysis takes L frames, a real-time estimate at frame t can use CES values up to frame t − 1 − L only, while the fast channels of frame t are in hand. The thinned replay (a sample visible every k-th frame) is a different condition: sparse, not delayed.

## 2. Seen before

The thinned replay (`docs/paper/b30_thinned.md`): with a sample every 50 ms the frozen network has +0.108 to +0.129 over the causal GP and the fast-channel increment is +0.076 to +0.156; its skill over hold falls with sparser sampling. Not seen: any readout with a dense, delayed history.

## 3. Design

Cut population, T_i, the four file-level splits, test discharges. Networks: the causal variant (`data/.b29_norm0_cut_s{seed}`) and the history-only twin (`data/.b29a1_v2h_cut_s{seed}`), frozen.

Delay L in frames (10 ms), L = 0, 1, 2, 4. At grid row t a CES sample measured at row u is available if u ≤ t − 1 − L (L = 0 is the present protocol). The CES-derived input channels of the networks (last available value, its staleness, the observed flag, for T_i and for rotation) are rebuilt from the available samples only, with the true staleness (the estimator is told how old the sample is); the fast channels and the time step are unchanged. Reported beside it: the variant in which the staleness is left as if there were no delay. Baselines on the same available samples: hold (the last available value) and the causal GP (`baselines_interpolation.predict_gp_causal`, the 16 last available samples, predicting at the stamp of row t). Scored rows: a measured T_i at row t and at least one available sample in the segment; the same rows for every estimator at a given L.

Check before the readout: at L = 0 the rebuilt inputs equal those of `seq_data.build_blocks`, and the estimates of the causal variant, hold and the GP agree with the frozen row table within 0.5 eV on at least 99 % of the rows of five test files per split; otherwise the run stops.

Intervals: 95 %, 2,000 resamples of test discharges; a statement holds if it holds on at least three of the four splits.

## 4. Statements

- **T1.** At L = 1 and at L = 2 the causal variant has skill over the causal GP with the interval above zero (each delay separately).
- **T2.** At L = 2 the causal variant has skill over the history-only twin with the interval above zero.
- Reported: for every L the root-mean-square error of hold, the GP, the twin and the causal variant; the skill of the causal variant over each; the fast-channel increment (causal variant over twin) and its difference from L = 0; the median absolute error of each; the same without the true staleness; L = 4.

## 5. What goes into the paper

- T1 and T2 hold: the paper states the use it serves (a CES analysis that reports one to three frames late is bridged by the fast channels) with these numbers, marked as frozen estimators trained without delay. Either fails: the paper says that the estimator is shown for a CES value available within one frame only.

## 6. Expectations

Root-mean-square error of the causal variant 8 to 25 % larger at L = 2 than at L = 0. Skill over the GP +0.10 to +0.22 at L = 1 and +0.12 to +0.28 at L = 2 (T1 holds). Increment over the twin +0.06 to +0.16 at L = 2 (T2 holds), larger than at L = 0 on at least three splits. Skill over hold lower at L = 2 than at L = 0.

## 7. Limits known in advance

The networks were trained with the previous frame available; staleness values of 20 to 50 ms occurred in training only at gaps. Training with an imposed delay is a GPU batch and is not part of this readout. A real analysis delay may vary from frame to frame.
