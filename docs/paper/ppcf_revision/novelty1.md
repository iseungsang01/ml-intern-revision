# Simulated PPCF novelty referee report (not an editorial decision)

**Recommendation: weak reject as submitted.** If the defects in change 1 are fixed, rewriting alone could reasonably bring this to borderline. I don't see rewriting alone reaching weak accept, for the reasons under "Honest statement" below.

## 1. Established components (the authors do not claim these)

The manuscript itself lists its methods as established (§1, "What is established…"): Matérn-3/2 GP regression, gradient boosting (scikit-learn defaults, §3.2, Supp. S2), LSTM and TCN networks, blocked validation and cluster bootstrap.

The closest prior work falls into three groups:

- **Readers of the raw spectrum.** Neural-network CES analysis dates back to JET, where it was presented as 
well suited to the fast intershot analysis of large volumes of data, and can readily be implemented in dedicated hardware for real-time applications
 (https://www.semanticscholar.org/paper/2316b88dbcfa394cb9fb58602979603c43192516). It has since been repeated on KSTAR [5] and on HL-3, where the authors propose 
a neural network-based method for real-time charge exchange recombination spectroscopy (CXRS) analysis on the HL-3 tokamak
 (https://www.fusionenergybase.com/publications?venue=Nuclear+Fusion).
- **Ti virtual diagnostics without CES.** RTCAKENN approximates 
carbon ion impurity temperature and rotation profiles, using real-time available inputs
 and 
executes in under 8 ms in the real-time environment
 (https://iopscience.iop.org/article/10.1088/1741-4326/ad142f).
- **Multi-rate observers and gap filling** [24–26, 15–18].

Taken together, "estimate a slow or delayed Ti from faster signals in real time" is not new. The manuscript concedes this (§1).

## 2. Application-specific evidence (the real content)

1. **Nowcast vs interpolation at D10 (Table 4).** TX beats PCHIP under all four designs; the weakest case is V4 at 15.3 [6.5, 22.7] %. Against the offline GP, TX wins only under V1 and V2 (V2: 3.7 [0.7, 7.0] %); the V3 and V4 intervals contain zero. T0 is worse than the offline GP everywhere (−3.1 to −5.6 %). The finding that "the fast channels carry the difference" is therefore supported. The median-error ratio of 1.06 against the offline GP (Table 4, lower block; S36 I11) is reported honestly.
2. **Fast-channel increment at D10 is small.** On F it is 1.9 [0.1, 3.6] % under V4, and on L5 it is not established under V3.
3. **Stale or sparse CES (Tables 5, 6, S8, S25).** This is the strongest evidence. At D650 the increment is 19–34 % across all designs. At K10, TX matches T0 with roughly 47 ms spacing. Controls support it: permutation (S17), anchor-row (S17), beam scalars (S16, S20).
4. **Bounds on the claims.** The increment is not shown to be specific to the radius: Mirnov carries about a quarter of it (S19). Under the S schedule it fails for shots ≥32312, including under the random design (Table 8). The fit must match the condition of use (S9, S10).
5. **MAST (Table 6, S21–S24).** Supports direction only, as the authors state.
6. **Runtime.** Median 1.95 ms, p99 4.59 ms (S13). This is estimator time only.

## 3. Genuinely supported distinct contribution

The paper offers a pre-registered, cluster-bootstrapped measurement, for one KSTAR pedestal-top chord, of how much collocated BES/ECEI add to a causal CES-history estimate. It covers the nowcast and stale/sparse availability rules across four validation designs, keeps negative outcomes (S33–S36; 47 of 170 statements fail), and compares against interpolations that wait for the next value. The novelty is evidential and methodological, not algorithmic.

Its practical value under the nowcast condition is modest:
- Two-thirds of estimates lie within 10 % of the level, short of the 10 % requirement (§5).
- Under the beam-modulation schedule, RMSE merely equals a single reading (Table 5, row S).
- The reading beats the estimate on the median (Table 7).

## 4. TCN evidence, assessed separately

The abstract and §1 make claims about a "separately audited" TCN. However, every pointer to it is an unresolved reference ("§??", in §1, §3.2, §3.7 and §4.1). Neither PDF contains a TCN section, table, latency row or seed count. So I cannot verify:
- the claim that "with the previous frame visible its fast-enabled form is not established better… and is worse in time order";
- the claim that its stale-history gain holds "under the random design only in both of its evaluation scopes";
- any distinction between a one-seed full-scope evaluation and a three-seed subset.

The paper also reuses the label TX for the fast TCN in its result files (§3.2), which invites confusion.

**As submitted, the TCN evidence carries zero weight.** The GP+boosting conclusions do not depend on it. The negative abstract sentence about the TCN must either be backed by a table or removed.

## 5. Ranked changes

**1. Restore the TCN section and fix broken references (rewrite-only, if the audited results exist).**
- Add a table giving the TCN F/L5 increments for each design, with the scope stated explicitly: full scope with one seed vs a subset with three seeds, and per-seed spread.
- Resolve the "[? ]" citations throughout the Supplement (S1, Table S1).
- *Falsifier:* if the tabulated three-seed subset shows the fast TCN established above the history TCN at D10 under V3 or V4, the abstract's "is not adopted / worse in time order" sentence must be withdrawn.

**2. Quantify the operational claim against the requirement (new analysis on existing fits; CPU only).**
- Report the TX share within 10 % of L5, with intervals, stratified by ELM-burst and notch proximity and per design.
- Report the nowcast–offline-GP difference restricted to "moving" frames, where the paper says the gain lies.
- *Falsifier:* if, on moving frames under V3 and V4, TX does not beat the offline GP (interval excludes zero in TX's favour), the statement "the gain is in the frames in which the value moves" is unsupported in time order.

**3. Radial specificity and the last shot block (needs new measurements).**
- Obtain the per-shot B_T, ECEI LO frequency and an equilibrium for the chord. Then repeat S19 and Table 8 with channels selected by mapped radius.
- *Falsifier:* if radius-matched channels give no larger increment than Mirnov or off-radius channels, the "collocated fast diagnostics" framing reduces to "any fast plasma signal".
- No GPU or new fits beyond CPU-scale boosting are needed except for any TCN reruns.

## Honest statement

Rewriting alone (change 1, plus tightening the abstract) can reasonably reach borderline. It cannot fix the core novelty limits:
- one chord and one campaign;
- the nowcast advantage over the two-sided GP is not established in time order;
- radial specificity is unproven;
- the S-schedule fails on the last block.

These need the measurements in change 3 or an independent campaign, not editing.