# B.47 — after B.39: the ablation with three initialisations, an estimator trained with a delayed CES history, and the inclusive population

Registered 2026-11-18, before any run of this batch and before the pooled arm of B.39 was read. GPU batch; training is never run on a CPU. It starts when the B.39 batch has ended. Designed by the session lead; the runner and the readouts are implemented by a delegated agent from this document, and no shared training code is edited while B.39 runs.

## 1. Why

Three statements of the restructured manuscript rest on less than its main table does.

1. "The increment is carried by the beam-emission channels" (B.39 arm A, B.45) rests on one initialisation per ablation network.
2. "Both margins remain when the CES value arrives late" (B.42) is a statement about frozen estimators trained without delay; the manuscript lists that as a limit.
3. The main text is written for the population without values above 3 keV. The readouts with those values kept come from an earlier, asymmetric pairing (30-epoch cap for one network, whole-discharge standardization). The registration of B.39 names the inclusive population as an extension with the same arms and statements.

## 2. Seen before

B.39 arm A, initialisation = split seed (splits 42, 1, 7, 123), over the history-only twin: BES-only +0.074, +0.026, +0.105, +0.063 (four of four intervals above zero); ECEI-only +0.010, +0.001, +0.084, +0.038 (one of four); the full estimator over BES-only −0.003, −0.004, +0.021, −0.003 (every interval includes zero). B.45: the BES-only increment in quiet rows +0.089, +0.016, +0.112, +0.065 (three of four). B.42 (frozen, 30-epoch cap, trained without delay): at a delay of two frames +0.075 to +0.161 over the causal GP and +0.043 to +0.134 over the twin (four of four each); the staleness channel hardly matters to the estimator (−0.006 to +0.005). Inclusive population, earlier runs: the causal variant under a 30-epoch cap +0.098 to +0.165 over the causal GP; the increment of the whole-discharge variant +0.016 to +0.024 (two of four). Not seen: any ablation network at a second initialisation; any network trained with a delayed history; the causal variant and its twin in the inclusive population under the same cap.

## 3. Arms, in the order of the runs

The environment of every run is that of B.39 (`run_b39.py`: the confirmed protocol through `run_b29.run_env`, every `CES_*` variable pinned, epoch cap 80, fast channels standardized with the statistics of the training discharges). Four file-level splits (`data/.b1_w2cut_split_s{seed}`, seeds 42, 1, 7, 123). Run directories `data/.b47_*` only.

**Arm D — ablation, two further initialisations (24 runs).** `v2e` (ECEI and the CES history), `v2b` (BES and the CES history), `v2n` (no Mirnov), each at the two further initialisations of its split as defined in B.39 (`run_b39.inits(seed)[1:]`). Cut population. With the runs of B.39 this gives three initialisations for every ablation network, for the full estimator and for the twin.

**Arm L — trained with a delayed CES history (16 runs).** The causal variant (`v2`) and the history-only twin (`v2h`), initialisation = split seed, cut population, trained with the six CES-derived input channels built under a delay of L = 2 frames by the rule of B.42 (a value measured at row u is available at row t only if u ≤ t − 1 − L; the staleness is the true age). The delay is applied in the training process by replacing the block builder of that process; `seq_data.py`, `train_seq.py` and `eval_seq.py` are not edited. Each network is trained twice: with L = 2 (the arm) and, as the control of the wrapper, with L = 0 through the same wrapper (8 + 8 runs); the L = 0 control must reproduce the test error of the corresponding B.39 run (`data/.b39_v2_cut_s{seed}_i{seed}`, `data/.b29a1_v2h_cut_s{seed}`) within run-to-run noise, and is reported.

**Arm I — inclusive population (24 runs).** The causal variant and the history-only twin with values above 3 keV kept (`CES_TI_SPIKE_CUT_EV=0` and the inclusive split and scoring of B.29), three initialisations per split as in B.39.

Scoring. Arms D and I: `eval_seq.py` on the test rows of the split, as in B.39. Arm L: the row machinery of B.42 (`b42_delay/run_b42.py`), which rebuilds the delayed inputs, hold and the causal GP from the available values and scores every estimator on the same rows, at L = 0, 1, 2, 4; before the readout it must reproduce the frozen-network numbers of `docs/paper/b42_delay.md` when pointed at the networks of B.42.

## 4. Statements

Four splits; a statement holds if it holds on at least three. Intervals: 95 %, 2,000 draws; where three initialisations exist a draw takes one initialisation of each network uniformly and resamples the test discharges (the rule of B.39).

- **D1′.** The ECEI-only network has skill over the history-only twin with the interval above zero.
- **D2′.** The BES-only network has skill over the twin with the interval above zero.
- **D3′.** The full estimator has skill over the BES-only network with the interval above zero.
- **L1.** At a delay of two frames the estimator trained with that delay has skill over the causal GP on the same available values, interval above zero (2,000 resamples of discharges).
- **L2.** At a delay of two frames it has skill over the twin trained with that delay, interval above zero.
- **L3.** At a delay of two frames it has skill over the estimator trained without delay (`data/.b39_v2_cut_s{seed}_i{seed}`, read under the same delay), interval above zero.
- **S1i, S2i.** Inclusive population: the causal variant over the causal GP, and over the twin, as S1 and S2 of B.39.

Reported: the no-Mirnov network over the twin and the full estimator over it; the spread of each network between its initialisations; the statements R1 to R3 of B.45 read again with three initialisations; for arm L every pair at L = 0, 1 and 4 and the median absolute errors; for arm I the skills over hold, PCHIP and the offline GP and the median absolute errors; epochs run and whether a run reached the cap.

## 5. What goes into the paper

- D2′ holds and D1′ fails: the sentence on the beam-emission channels stands and the limit "one initialisation per ablation network" is removed. D1′ holds as well: both groups carry the increment and the sentence is changed. D2′ fails: the attribution to BES is withdrawn and stated as not established across initialisations. D3′ is stated as it comes out.
- L1 and L2 hold: the paper states that an estimator trained with a delayed history keeps both margins, and the limit on frozen estimators is removed for the delay. Either fails: stated, and the claim of use is bounded to a CES value in hand within one frame plus the frozen replay. L3 decides one sentence: whether training with the delay is needed.
- S1i and S2i are reported in the main text beside the cut numbers, whichever way they come out.
- Failed statements are reported as failed.

## 6. Expectations

D1′: +0.00 to +0.05 on three splits and up to +0.09 on the third; holds on one or two splits (does not hold). D2′: +0.03 to +0.10; holds on four. D3′: within 0.02 of zero; does not hold. Spread between initialisations of an ablation network (largest minus smallest skill over the twin): 0.01 to 0.04. L1: +0.07 to +0.16, holds. L2: +0.04 to +0.13, holds. L3: −0.01 to +0.03, holds on at most two splits (does not hold). L = 0 control: within 0.02 in skill over the causal GP of the B.39 run. S1i: +0.09 to +0.17, holds on four. S2i: +0.01 to +0.04, holds on two or three splits.

## 7. Limits known in advance

The splits and discharges are those of every earlier readout; no untouched hold-out exists. One initialisation in arm L. A fixed delay of two frames; a real analysis delay may vary. The delay is imposed on the inputs; the loss and the scored rows are unchanged, so rows whose only available value is older than the block start are scored by the readout rule of B.42. Arm I is run last and may be cut short; it is then reported for the runs that are complete, by split.
