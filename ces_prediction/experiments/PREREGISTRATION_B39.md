# B.39 — the causal variant as the primary estimator under a symmetric protocol, and which diagnostic carries the fast-channel increment

Registered 2026-11-17, before any run. GPU batch; training is never run on a CPU.

## 1. Why

Two points of the tenth independent grading of the PPCF draft (manuscript only; `PROJECT_KNOWLEDGE.md` 4.13), both of which `PAPER_DESIGN.md` section 10 already lists as differences between the fixed design and the manuscript.

1. The title says "causal estimator", but the primary readout (pooled cross-validation) is that of the network whose fast channels are standardized with whole-discharge statistics. The causal variant has no pooled readout; its fast-channel increment quoted in the abstract (+0.027 to +0.115) is a pairing made after the fact between runs with different epoch caps (30 and 80); on the file-level splits every configuration was trained once, so the spread between initialisations is not in any interval.
2. No network was ever trained with a subset of the fast channels. BES is emission of the heating beam and the CES fit depends on the beam; a network that reads ECEI only (independent of the beam) is the direct test of whether the increment rests on a beam-dependent signal. The Mirnov channels are described as carrying no information and are inputs all the same.

## 2. Seen before

The causal variant under a cap of 30 on the four file-level splits (over the causal GP +0.068 to +0.141; over the history-only twin +0.027 to +0.115, post hoc). The whole-discharge variant: pooled over the causal GP +0.113 [+0.085, +0.132]; over its twin under a cap of 80, +0.064 to +0.117. The 20-parameter filter (summed gains +48 to +53 eV per standard deviation for BES, +44 to +50 for ECEI). B.34 (the error model uses the BES level and the ECEI level). B.38 (training losses of out-of-fold causal-variant networks). Not seen: any causal-variant run under a cap of 80, any pooled causal-variant or pooled twin run, any network trained on a subset of the fast channels.

## 3. Design

Cut population (values above 3 keV treated as missing for every estimator). T_i. The environment is that of the B.29 runs (`run_b29.run_env`: confirmed protocol, held-free, every `CES_*` variable pinned) with an epoch cap of 80 for every network; networks that read fast channels standardize them with the statistics of the training discharges (`CES_PER_SHOT_NORM=0`). The history-only twin reads no fast channel, so their standardization does not enter it.

**Arm A, four file-level splits** (`data/.b1_w2cut_split_s{seed}`, seeds 42, 1, 7, 123).

- Causal variant (`v2`) and history-only twin (`v2h`): three initialisations per split, the split seed and the next two in the cycle (42, 1, 7, 123). The twin run with initialisation = split seed exists (`data/.b29a1_v2h_cut_s{seed}`, cap 80) and is reused.
- Three new variants of the T_i encoder, one initialisation (the split seed), everything else as the causal variant: `v2e` reads the 4 ECEI channels and the 7 CES-derived channels; `v2b` the 9 BES channels and the 7; `v2n` BES and ECEI without the 2 Mirnov channels. The channels are removed from the encoder input, not set to zero. The rotation branch is unchanged in all.

**Arm P, pooled cross-validation.** The five folds of repeats r2 to r9 of B.27 (`data/.b27_split_r{r}_f{k}`, the partitions of the manuscript's pooled readout; initialisation seed 27000 + 10 r + k as there): the causal variant and the history-only twin, 40 runs each.

Scoring: `eval_seq.py` on the test rows of each split (the harness of every earlier readout); skill = 1 − sum of squared errors of A over that of R on the same rows.

Order of the runs (so that an interrupted batch still answers the first questions): arm A with initialisation = split seed (causal variant, the three ablations), then the remaining initialisations of arm A, then arm P.

## 4. Statements

Four splits; a statement holds if it holds on at least three of four.

- **S1.** Causal variant over the causal GP: the interval is above zero. Interval: 2,000 draws, each one of the three initialisations chosen uniformly and the test discharges resampled with replacement.
- **S2.** Causal variant over the history-only twin: the same two-level interval (an initialisation of each network drawn independently) is above zero.
- **D1.** The ECEI-only variant over the twin (initialisation = split seed for both; 2,000 resamples of discharges): interval above zero.
- **D2.** The BES-only variant over the twin: interval above zero.

Pooled; the two-level interval of the manuscript's pooled readout (20,000 draws: a repeat, then its discharges).

- **P1.** Causal variant over the causal GP: interval above zero.
- **P2.** Causal variant over the history-only twin: interval above zero.

Reported: the skills over hold, PCHIP and the offline GP; the skill of each initialisation and the spread within a split; the full causal variant over the ECEI-only, BES-only and no-Mirnov variants, and the no-Mirnov variant over the twin; the median absolute error of every network beside that of hold; the fold-level interval with the Nadeau–Bengio correction for P1 and P2; epochs run.

## 5. Decision

- P1 and S1 hold: the causal variant becomes the estimator of the manuscript's primary table and the whole-discharge variant a sensitivity row; "causal" stays in the title.
- P2 and S2 hold: the fast-channel increment is stated for the causal variant from this pairing and the post hoc pairing is removed. Either fails: the increment of the causal variant is stated as not established under the symmetric protocol.
- D1 holds: the increment does not rest on a beam-dependent signal. D1 fails and D2 holds: it is stated as resting on the beam emission, and the beam is kept as the first caveat. Both hold: both groups carry it. Neither: the increment needs the combination and cannot be attributed to a group by removal.
- Failed statements are reported as failed.

## 6. Expectations

S1: +0.07 to +0.14 per split (three initialisations pooled), four of four. S2: +0.03 to +0.11, three or four of four. Spread of the skill over the causal GP between initialisations of one split (largest minus smallest): 0.01 to 0.04. D1: +0.01 to +0.06, two to four of four. D2: +0.02 to +0.08, three or four. Full over no-Mirnov: within 0.015 of zero. P1: +0.08 to +0.12, holds. P2: +0.04 to +0.09, holds.

## 7. Limits known in advance

The recurrent design and the standardization were adopted on these four splits, and the pooled folds re-partition the same discharges; no untouched KSTAR hold-out exists. Cut population only: the inclusive population is an extension (the same arms and statements, run after this batch if the GPU stays free, reported separately). One initialisation for the ablations; a difference between ablations smaller than the spread between initialisations of arm A is not interpretable. An ablation by removal shows what a network can do without a group, not what the full network uses.
