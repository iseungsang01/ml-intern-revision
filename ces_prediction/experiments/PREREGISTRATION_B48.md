# B.48 — the row-level readouts of the manuscript repeated with the estimator of B.39

Registered 2026-11-18, before any of the readouts below was computed. CPU only; nothing is trained, frozen networks are run forward. Designed by the session lead and executed by a delegated agent from this document.

## 1. Why

The restructured manuscript takes its main table from B.39 (the causal variant and its history-only twin, both under an 80-epoch cap). Its sections on where the gain is and on a late or sparse CES value take their numbers from readouts made earlier with the causal variant of B.29 (`data/.b29_norm0_cut_s{seed}`, a 30-epoch cap) paired with the twin under an 80-epoch cap. The manuscript lists that as a limit. The same readouts with the estimator of B.39 remove it.

## 2. Seen before

With the 30-epoch estimator. B.41: the plain rows hold 0.910, 0.692, 0.602, 0.593 of the reduction against the twin (G1 holds on 1 of 4) and 0.690, 0.295, 0.417, 0.366 against the causal GP (G2 on 0 of 4); the increment inside plain rows +0.044 to +0.152 (four of four); the 90th percentile of the absolute error 0.920 to 0.943 of that of the causal GP and 0.765 to 0.809 of that of hold (Q1 holds); rows 1 to 3 after an absent frame +0.195 to +0.256 over the twin. B.42: T1 and T2 hold (L = 2: +0.075 to +0.161 over the causal GP, +0.043 to +0.134 over the twin). B.44: W1 holds on 1 of 4 and W2 on 2 of 4. B.30 at a visible value every 50 ms: +0.070 to +0.134 over the causal GP (four of four), +0.054 to +0.156 over the twin (four of four, after the fact), +0.066 to +0.127 over PCHIP of the visible values (three of four, after the fact). With the estimator of B.39 (initialisation = split seed) on all scored rows of `eval_seq`: +0.081, +0.094, +0.144, +0.101 over the causal GP and +0.071, +0.022, +0.123, +0.060 over the twin. Not seen: any row-level readout of the estimator of B.39 beyond those of B.45.

## 3. Design

T_i, cut population, the four file-level splits, test discharges. Estimator: the causal variant of B.39 with initialisation = split seed, `data/.b39_v2_cut_s{seed}_i{seed}`. Twin: `data/.b29a1_v2h_cut_s{seed}` (the `hist80` column of the frozen row tables). Hold, the causal GP and PCHIP as in the earlier readouts.

A row table with the columns of `data/.b30/rows_cut_s{seed}.npz` is built for the new estimator by the machinery of B.30 (`b30_thinned/run_b30.py`), at k = 1 and k = 5, in `data/.b48/`; every column except the estimator's is taken from, or must equal, the frozen table.

Checks before the readout (the run stops if one fails): (a) hold, the causal GP and the twin of the new table equal those of the frozen table on every row; (b) at k = 1 the squared error of the new estimator agrees with `comparison_errors_test.npz` of its run on the rows that both score, within 0.5 eV in absolute error on at least 99 % of them; (c) the delayed-input machinery of B.42 at L = 0 reproduces the k = 1 column within 0.5 eV on at least 99 % of the rows.

The readouts, each with the definitions, rows, intervals and rules of its registration, the only change being the estimator:

- from B.41 (`PREREGISTRATION_B41.md`): G1, G2, the increment inside plain and inside flagged rows, Q1 and the percentile table, the mean signed error by third and the slope, the blip-conditioned traces and the skills in rows 1 to 3 and 4 to 10, and the figure (written to `docs/paper/b48_events_fig.{pdf,png,json}`; the burst panel uses the new estimator on the raw-stream test discharges);
- from B.42 (`PREREGISTRATION_B42.md`): T1, T2 and the tables at L = 0, 1, 2, 4;
- from B.44 (`PREREGISTRATION_B44.md`): W1, W2 and the table by third of the median T_i (the other strata are reported);
- from B.30 at k = 5: the estimator over the causal GP, over the twin and over PCHIP of the visible values.

## 4. Statements

The registered statements are those of the earlier documents, read with the new estimator: **G1, G2, Q1** (B.41), **T1, T2** (B.42), **W1, W2** (B.44), and two for the sparse replay, on four splits with the rule "at least three":

- **H1.** With a value visible every fifth frame the estimator has skill over the causal GP with the interval above zero.
- **H2.** The same over the twin.

## 5. What goes into the paper

The numbers of the sections on where the gain is and on a late or sparse value are replaced by these, and the limit on the 30-epoch estimator is removed. A statement whose verdict differs from the earlier readout is reported with both readouts.

## 6. Expectations

The verdicts are those of the earlier readouts: G1 and G2 do not hold, Q1 holds, T1 and T2 hold, W1 and W2 do not hold, H1 and H2 hold. Increments over the twin within 0.03 of the earlier ones on each split; the 90th-percentile ratio to the causal GP 0.90 to 0.95; the skill over the causal GP at L = 2 +0.07 to +0.17.

## 7. Limits known in advance

One initialisation (the split seed). The test rows of these splits have been read by every earlier readout, and the definitions were fixed with the earlier estimator. The frozen estimator was trained without delay and at the natural cadence.
