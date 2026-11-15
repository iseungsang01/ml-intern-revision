# PREREGISTRATION B.29 addendum 1 — is the cut-population fast-channel increment an effect of the 30-epoch cap

Registered 2026-11-15, before any addendum run and before any addendum code is executed on a GPU.
Committed first; the code (`b29_causal/run_b29.py --addendum1`, `b29_causal/analyze_b29_addendum1.py`,
the test) follows in a separate commit. Parent registration: `PREREGISTRATION_B29.md`.

## 1. Why

Arm A of B.29 (history-only `seq_v2`, 8 runs) is finished and read (`docs/paper/b29_causal.md`,
record at the end of `THESIS_RESULTS.md` section 8, commit bd08598). **Its TEST results have been
seen**: T_i backbone vs history-only on the four cut splits +0.067 [+0.029, +0.108],
+0.077 [+0.047, +0.106], +0.117 [+0.093, +0.142], +0.059 [+0.011, +0.103] (4/4 exclude 0), inclusive
+0.016 to +0.024 (2/4). H29-1 was read as "title kept, increment conditional on the cut".

The registered epoch cap is 30 (the backbone's). In the cut population the history-only arm reached
it on 3 of 4 splits:

| split | arm A cut: epochs run (best epoch) | frozen backbone cut: epochs run (best epoch) |
|---|---|---|
| 42 | 30 (29) | 25 (19) |
| 1 | 30 (30) | 20 (14) |
| 7 | 11 (5) | 20 (14) |
| 123 | 30 (30) | 30 (29) |

An arm whose validation error was still falling at the cap may be undertrained, which would
overstate the increment. The backbone reached the cap on one split (123), so it is not exempt.
The inclusive arms all stopped early (best epoch 7 to 17) and are not part of this addendum.

## 2. Design — one variable

Epoch cap 30 -> **80**. Everything else is the parent registration: early stopping on validation
MSE with patience 6, `ReduceLROnPlateau` (patience 2, factor 0.5), MSE loss, batch 16, split seed =
init seed, `CES_PER_SHOT_NORM=1`, `CES_SEQ_NORM_EXPANDING=0`, `CES_TI_SPIKE_CUT_EV=3000`, the
`GATE_ENV` harness, the frozen split manifests and harness statistics. Every `CES_*` variable is
stripped and pinned by the runner. The cap is not read by the scheduler or the optimiser, so a run
that stops early under cap 30 follows the same procedure under cap 80.

Runs (cut population, four file-level splits, 8 in total):

| run | model | output | paired against |
|---|---|---|---|
| arm A, cap 80 | `v2h` | `data/.b29a1_v2h_cut_s{s}` | frozen backbone; cap-80 backbone |
| backbone, cap 80 | `v2` | `data/.b29a1_v2_cut_s{s}` | frozen backbone; cap-80 arm A |

The frozen backbone (`data/.b1_seqv2_s{s}_i{s}`) and the first arm A runs (`data/.b29_v2h_cut_s{s}`)
are not modified and stay on record. The backbone is retrained on all four splits, not only on 123,
so that both sides of read-out (ii) were produced by one code path on one day. For splits 42, 1 and
7 the frozen backbone stopped before epoch 30, so its cap-80 retrain is a replicate that differs
only by GPU non-determinism; the agreement of the two is reported as the run-to-run noise of the
comparison.

Identity check before any statistic (in the analysis, fatal on mismatch): the cap-80 backbone's
`metrics.json` must equal the frozen backbone's on `seq_model`, `n_params`, `per_shot_input_norm`,
`feature_dims`, `normalization`, `train_blocks`, `train_labels`, `drop_stuck_targets`,
`fixed_epochs`, `seed`, `init_seed`, `loss`; and `paired_model_compare.py` verifies the TEST rows
row for row.

## 3. Read-outs (T_i, cut, per split; shot-clustered paired bootstrap as in the parent)

- (i) frozen backbone vs cap-80 arm A — the parent's cell with only the arm retrained.
- (ii) cap-80 backbone vs cap-80 arm A — **primary for this addendum**.
- (iii) cap-80 arm A vs first arm A, and cap-80 backbone vs frozen backbone — what the longer cap
  changed on each side.
- cap-80 arm A vs causal GP (the H29-1b cell), epochs run, best epoch, whether cap 80 bound.
- Guard: V_rot (ii) is expected to include 0 on 4/4.

## 4. Rules (fixed now)

- **R-a.** If the lower bound of (ii) is > 0 on >= 3 of 4 cut splits, the fast-channel increment
  on the cut population is quoted as final, with the (ii) numbers, and the first-run numbers go to
  the record as the registered primary read at cap 30.
- **R-b.** If (ii) has a lower bound > 0 on fewer than 3 of 4 splits, the paper says that the cut
  increment is sensitive to training length, quotes both the cap-30 and the cap-80 numbers, and the
  title rule is re-read with the cap-80 numbers: H29-1 fails, so the title changes to
  noise-limited estimation of the CES past as the parent registration prescribes.
- **R-c.** If (i) and (ii) disagree on the 3-of-4 count, (ii) decides and the disagreement is
  stated with the replicate noise of (iii).
- **R-d.** If a run still reaches cap 80 (epochs run = 80), that is reported for the run and no
  further extension is made.
- The inclusive statement ("conditional on the cut", 2/4) is not touched by this addendum.

## 5. Numerical expectations (stated so they can fail)

- Arm A improves with the longer cap on splits 42, 1 and 123: cap-80 arm A vs first arm A between
  +0.005 and +0.04 in T_i skill; split 7 (early-stopped at epoch 5) changes by less than 0.01 in
  absolute value.
- The increment (ii) shrinks to +0.03 to +0.10 per split and its lower bound stays > 0 on 3 or 4
  of 4 splits (R-a).
- Cap-80 backbone vs frozen backbone: within +-0.01 on splits 42, 1, 7 (replicates); 0 to +0.03 on
  split 123.
- No run reaches epoch 80; arm A stops between epoch 30 and 60 on the three capped splits.
- Cap-80 arm A vs causal GP excludes 0 on 2 to 4 of 4 splits (first run: 2/4).

## 6. Disclosures

- Post hoc: the cap was questioned after the arm A TEST table was read. The parent registration
  (section 3, tables) only asked that cap binding be reported.
- 8 GPU runs, about 1.5 to 3 h at the measured 7 to 16 min per 30-epoch run; never on CPU.
- Output: `docs/paper/b29_addendum1.json` / `.md`, record in `THESIS_RESULTS.md` section 8.
