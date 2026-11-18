# B.67 — weighing the sources before the recurrent layer: source-gated fusion

Registered 2026-11-19, before any run of this batch. GPU. The session lead designed the batch. An AI agent implements and runs it under the author's supervision. No existing file is edited: the new architecture is added to the registry at run time by wrapper scripts. The batch runs after B.65.

## 1. Why

The T_i encoder of the paper's network reads the 22 inputs of a row as one vector. Inside its first LSTM layer every input gets a fixed weight, the same at every row. Attention-style models instead compute, at every row, how much each source counts. The author asked whether that would do better. The fourteenth grading also named a missing mechanism as a main weakness. Per-row source weights are a readable quantity: they show when the network leans on the beam-emission channels and when on the CES history.

## 2. Seen before

- **Temporal attention.** Attention over time was tried (§8ag). At contexts of 150 ms and more, LSTM, TCN and attention ended within 0.023 of each other. Attention across sources within a row has not been tried.
- **Pooled B.39, r2 to r4.** The network over the twin: +0.053 [+0.027, +0.076].
- **B.59, inputs that carry the increment.** BES with the CES history carries it: +0.044 [+0.020, +0.066] over the twin. ECEI with the history carries none. A network without Mirnov is at least as good: +0.060 over the twin.
- **Where the increment is.** Rows 1 to 3 after an absent frame carry most of it (B.59: +0.153 over the twin).

## 3. Design

**Folds and environment.** The 15 folds of re-partitions r2, r3 and r4 of the pooled cross-validation of B.39. The environment of every run is `run_b39.pooled_env` of the fold:

- cut population;
- cap 80 with early stopping;
- `CES_PER_SHOT_NORM=0`;
- the seeds and statistics of the fold.

The only change is `CES_SEQ_MODEL` and the output directory (`data/.b67_{S,A}_cut_r{r}_f{k}`). There is one initialisation per fold, as in B.39 and B.59.

**Arms (15 runs each, 30 in all).** Every arm differs from `seq_v2` in one thing: the layer between the inputs of a row and the T_i LSTM. The source groups of a row are:

- BES (9 channels);
- ECEI (4);
- Mirnov (2);
- CES-derived (7: dt, and the carried value, staleness and has-observation flag of each of T_i and V_rot).

The two arms:

- **S, gated concatenation (`v2s`).** Each group s gets its own embedding e_s = LayerNorm(Linear(x_s → 32)). Gate logits come from the concatenated embeddings through Linear(128 → 32), GELU, Linear(32 → 4). The gates a = softmax(logits) are per row. The LSTM input is [4 a_1 e_1 | 4 a_2 e_2 | 4 a_3 e_3 | 4 a_4 e_4]. Uniform gates give the plain concatenation of the embeddings, so no source is cut off; the gates only reweigh.
- **A, attention pooling (`v2a`).** Same embeddings and gates, but the LSTM input is the weighted sum Σ_s a_s e_s (32 values). The row's information has to pass through the weights. This is the classical attention bottleneck.

**What stays the same.** In both arms the gate at row t uses inputs of row t only, so the model stays causal. The T_i LSTM (2 layers, 160 wide), its head, the V_rot branch (history only), the loss and the training are those of `seq_v2`. The parameter count stays below 1,000,000.

**Readouts.** These follow B.59 pooled: `analyze_b39.arm_p`, 20,000 draws, the stored B.39 errors of the network (`v2`) and the twin (`v2h`) on the same folds. Also reported:

- skill over the causal GP;
- median absolute error;
- rows 1 to 3 after an absent frame.

**Gate readout (arm S).** The mean gate of each source, saved on the test rows, by row class:

- rows 1 to 3 after an absent CES frame;
- rows right after a measured frame;
- rows within 30 ms of a burst onset, and inter-burst rows (B.66 detector and classes);
- the three temperature thirds of B.54.

Intervals for the gate readout resample discharges and draw the re-partition.

**Checks before any run.**

- On CPU: forward shapes and parameter counts.
- Causality: perturbing the inputs after row t leaves the outputs up to t unchanged.
- The wrapper registers the variants for both training and scoring.
- The 30 planned environments equal those of B.59 arm N apart from `CES_SEQ_MODEL` and the directories.
- No tracked file is changed.

## 4. Statements (r2 to r4 pooled, cut population)

- **G1.** S over the network (`v2`): the interval is above zero.
- **G2.** S over the twin: the interval is above zero.
- **G3.** A over the network: the interval is above zero.
- **G4.** In arm S the BES gate is higher in rows 1 to 3 after an absent frame than in rows right after a measured frame. The interval of the difference is above zero.

## 5. What goes into the paper

The estimator of the paper is not changed by this batch.

- **G1 or G3 met.** The paper states that weighing the sources per row raises the skill by the stated amount, and gives that variant as the next model for a confirmatory run.
- **Neither met.** The paper states that per-row source weighting, with or without a bottleneck, adds nothing over the fixed weights of the LSTM input layer. This is the answer to "would attention do better".
- **G4 met.** The gate figure goes into the section on what carries the increment. It is read as which source the trained network weighs, not as a physical cause.
- **G4 not met.** The gate readout goes to the supplement.

## 6. Expectations

- G1: −0.01 to +0.02, not met.
- G2: +0.03 to +0.06, met.
- G3: −0.03 to +0.01, not met.
- G4: met, since the history is stale in exactly those rows.

## 7. Limits known in advance

- There is one initialisation per fold and three re-partitions.
- The gates are a property of one trained network per fold. Gates of a softmax over four sources can stay near uniform and still let the embeddings carry the reweighing, so a flat gate does not show that the sources are used equally.

## Addendum 1 (2026-11-19, before any run)

Definitions the registration left open, fixed at implementation time and before any GPU run.

- **Rows right after a measured frame.** Not a beam-blip row, and the previous kept row is 10 ms earlier (age 10 ms within 0.5 ms) with a finite T_i, as in the age and hold variables of B.54. About 96 % of scored rows are in this class. It is checked row by row against the network's own staleness input.
- **Rows within 30 ms of a burst onset, and inter-burst rows.** These are `run_b66.phase_masks`: 0 to 30 ms after an onset, and at least 50 ms from every onset. They are taken within classes B and C of B.66, as in its statement R3. Classes A, B and C are also reported whole.
- **Re-running a failed readout.** An arm S run counts as done only when both its scored errors and its gate file exist. Otherwise it is scored again. A failed gate-alignment check marks the run as failed.
- **Descriptive readouts, without statements.** The following gate differences are reported but carry no statement:
  - burst rows minus inter-burst rows;
  - class C minus class A;
  - hottest third minus coldest third;
  - per-row gate quantiles.

  All gate intervals use the same draws, so each difference is formed draw by draw.
- **Checks.** All passed before any run: `docs/paper/b67_checks.md` lists checks (a) to (h).
  - The parameter counts are 430,758 (S) and 369,318 (A), against 357,570 for `seq_v2`.
  - Causality is bit-exact.
  - The gate alignment reproduces the stored errors of the network on all 37,483 rows of r2 f0, and a negative control shifted by one row fails.
  - The 30 environments equal those of B.59 arm N apart from the model and the directory.
