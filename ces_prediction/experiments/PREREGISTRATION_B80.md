# B.80 — the coherence of the ECEI channels as an input-only condition for the fast-channel increment

Registered 2026-11-20, before any run of this batch. CPU only; no fit: stored estimates of B.75 and B.79 and input-only quantities. The session lead designed it. An AI agent implements and runs it under the author's supervision. No existing file is edited.

## 1. Why

Under the beam-modulation schedule (rule S of B.75) the fast-channel increment holds in the shots up to 32310 and is not established from 32312 on, with or without beam inputs (B.75 R8, B.79 N7). An unregistered diagnostic (`docs/paper/b75_last_block.md`) found, without any estimator, that the ECEI of the last block is another signal: its level does not fall at a beam notch (1.001 of the level before, against 0.902 and 0.967 in the earlier blocks, while T_i and BES fall in all three), its four channels are less correlated with each other (0.654 against 0.911 and 0.841, median over discharges), and its change since the newest visible CES value does not go with the change of the ion-temperature level (Spearman -0.013 against +0.269 and +0.329).

That comparison is between blocks of shots, three points. This batch asks whether an input-only measure of the ECEI signal orders the increment **discharge by discharge, inside every block**. If it does, the limit of the proposal becomes a condition that can be checked on the inputs, and an estimator that uses the ECEI only where the condition is met can be read on the last block.

**Status.** Exploratory on used data. Known when this was written: the block medians above, the per-block increments of B.75 and B.79, and the per-discharge contributions of the last block listed in the diagnostic. Not known: the distribution of the coherence over discharges, and any increment by coherence.

## 2. Data

- Stored per-unit estimates of B.75 (`data/.b75/fits/`: T0, TX, TB, TE under D650 and S) and B.79 (`data/.b79/fits/`: TN, TXN), target L5, the units of B.73 (V1, V2, V3, V4 with fits to 31991 and to 32310). TB and TE exist for V1 and V4 only.
- Row sets: B.75's common set (D650) and S set (S).
- ECEI channels of the grid (`data/.b73/grid.npz`, the four columns after the nine BES columns).

## 3. The measures (inputs only)

Per discharge, over its grid rows (the first and last row of a file excluded), discharges with at least 50 grid rows:

- **C_E (coherence):** the mean of the six pairwise Pearson correlations between the four raw ECEI channels.
- **C_E early:** the same over the first 200 grid rows of the discharge (reported beside C_E: the share of discharges that fall into the same class; no statement uses it).
- **N_E (notch response):** the median, over the absent steps of the discharge that have five rows of the block before them, of the ECEI array mean in the first row after the step divided by its median over the five rows before; defined for discharges with at least three such steps. Reported; no statement uses it.

**Classes of C_E, fixed here:** low (below 0.7), middle (0.7 to 0.9), high (0.9 and above). A discharge without C_E belongs to no class and is left out of the class readouts (count reported).

C_E uses the whole discharge. It is a property of the inputs and reads no CES value, but it is not available at the first frame; C_E early says how much of it is.

## 4. Readouts

On identical rows, target L5, rules S and D650, per design: by class of C_E, the rows, discharges and clusters; RMSE of T0, TX; TX over T0; TXN over TN; TB over T0 and TE over T0 (V1, V4); the same within each chronological block (up to 31991, 32002 to 32310, from 32312) under V1; the distribution of C_E and N_E per block; the Spearman correlation over discharges between C_E and the per-discharge skill of TX over T0 (V1, S; discharges with at least 50 scored rows).

**Gated estimator (no fit).** TG = TX for the rows of discharges with C_E of 0.7 or above, T0 for the others (discharges without C_E use T0). TG over T0 and TG over TX per design and rule, and under V1 by block.

**Intervals.** 95 %, 20,000 draws, shot clusters resampled (the discharge form beside it), a re-partition drawn first under V1 and an assignment under V2, as B.75. A statement is judged on the cluster interval. Seed 20261180.

## 5. Statements (target L5)

| Id | Statement | Expectation |
|---|---|---|
| G1 | V1, S: TX over T0 in the high class minus that in the low class lies above zero | holds |
| G2 | V1, S: TE over T0 in the low class has an interval that contains zero or lies below it (the ECEI half adds nothing where its channels are not coherent) | holds |
| G3 | V1, S: TE over T0 in the high class lies above zero | holds |
| G4 | V1, D650: TX over T0 in the high class minus that in the low class lies above zero | holds |
| G5 | V1, S, blocks up to 32310 only: TX over T0 in the high class minus that in the low class lies above zero (the ordering holds inside the earlier blocks, not only between blocks) | uncertain, leaning to holds |
| G6 | V3, S: TX over T0 in the high class lies above zero | holds |
| G7 | V1, S, block from 32312: TG over T0 lies above zero | uncertain |
| G8 | V4 (fit to 31991), S: TG over T0 lies above zero | uncertain |
| G9 | V1, S, all blocks: TG is not worse than TX (the interval of TG over TX is not below zero) | holds |

Each statement is judged separately; the tables are reported whole.

## 6. What goes into the paper (decided now)

- If G1 and G2 hold: the limit is stated as a condition on the inputs: where the ECEI channels of a discharge are not coherent the ECEI adds nothing and the increment under the schedule is smaller, with the class values; the last block is reported with its share of low-coherence discharges. If G5 also holds, the text says the ordering holds inside the earlier blocks.
- If G1 fails: the last block is reported as a limit without an explanation, and the diagnostic stays a description.
- If G7 or G8 holds: the gated estimator is named as the form that uses the ECEI only where its channels are coherent, with the sizes; if neither holds, the gate is described and not proposed.
- G9 failing means the gate costs accuracy overall; then it is not proposed whatever G7 and G8 say.

## 7. Limits known in advance

- Exploratory on used data; the measure was chosen after the block comparison had been seen.
- C_E is computed over the whole discharge. No field, ECEI setting or channel radius is in the files, so what lowers the coherence is not decided here.
- The cuts 0.7 and 0.9 are fixed by hand before the distribution was seen; the tables by class show what other cuts would give only coarsely.
- TB and TE exist for V1 and V4 only; no new fit is made, so there is no estimator refitted within a class.

## 8. Checks and files

- The stored estimates used are those of the committed outputs of B.75 and B.79 (the increments of TX over T0 under S and D650 recomputed over all classes equal the published values of `b75_availability.md` to the third decimal).
- C_E reads only the ECEI columns (stated; nothing to test against).
- Code: `ces_prediction/experiments/b80_ecei_gate/`. Caches: `data/.b80/`. Output: `docs/paper/b80_ecei_gate.{md,json}`.
- Implementation choices left open go into `PREREGISTRATION_B80_ADDENDUM1.md`, written before any readout. This file is not edited after the first run.
