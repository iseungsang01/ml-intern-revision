# Data-set ledger: how every result of the paper selects and treats the data

Read-only audit of 2026-11-19 at commit `b7b2da4` (plus the untracked `docs/paper/b68_auxtarget.*`).
Source of record is the code (runner, analysis script, env dicts, loaders), not the prose; where the
prose disagrees, section 5 says so. Line numbers are those of that commit. Counts marked
"measured" were recomputed for this ledger from the stored artifacts under `data/` (CPU, read-only).

Scope: every batch of `PROJECT_KNOWLEDGE.md` (PK) section 4 from 4.26 on, B.42 (4.23), the
earlier 4.2x sections that `main_ppcf.tex` / `supp_ppcf.tex` still cite (B.40, B.41, B.43,
B.44, B.45, B.46), B.68 and B.69, and, in block D, the older batches the supplement still cites.

---

## 1. Shared machinery (the codes used in the ledger)

### 1.1 Discharge pools

| Code | Definition (code) | Count |
|---|---|---|
| F641 | every `data/s*.csv` | 641 |
| U639 | universe of the pooled cross-validation: train + val + test of `data/.b1_manifest_s42` (`b27_kfold/run_b27.py:53, 111-112`). Leaves out s30967 and s31702 (three CSV rows each; only one row carries a value) | 639 |
| S638 | the cut four-split manifests `data/.b1_w2cut_split_s{42,1,7,123}`: U639 minus s31331 (its single T_i is > 3 keV and its two V_rot values are a held repeat, so no row survives cut + held removal). train/val/test = 415/127/96 (s42), 414/128/96 (others) | 638 |
| I639 | the inclusive four-split manifests `data/.b1_manifest_s{s}`: 415/128/96 | 639 |
| Pooled fold | `run_b27.stage_folds` (`run_b27.py:110-137`): U639 permuted by `REPEAT_SEEDS[r]` = 2701+r, 5 folds, val = 128 (20 % of U639), train 383-384, test 127-128. r0-r1 = B.27, r2-r9 = B.27c. Train fraction 60 % (four splits: 65 %) | 639 |
| Scored (pooled) | discharges with >= 1 scored T_i row in a re-partition: 633-635 (measured). Never scored: s31002, s31331, s32501, s32570 (0-1 finite T_i) and the two outside U639; the 633-635 variation comes from single-row discharges (31767, 31482, 31483, 31485) that the 40,000 cap of 1.4 drops in some folds | 633-635 |
| TS | time split (`campaign/run_campaign_shift.py:78-109`, called by `b5_rescore/run_b5.py`): F641 sorted by shot, first int(0.65 x 641) = 416 train (30801-31991; 413 usable under the cut, 414 inclusive), 128 val (32002-32310), 97 test (32312-32751; 95 carry scored rows). Identical for all seeds | 641 |
| RAW12 | the 12 discharges with raw streams (`analyze_file_rules.py:43`, `RAW_SHOTS`) | 12 |

### 1.2 Loaders: edge rows, held values, values above 3 keV

| Loader | Used by | First/last rows (t = 0, 30 s) | Held values | T_i > 3 keV |
|---|---|---|---|---|
| L-seq `seq_data.load_grid_files` (`seq_data.py:119-172`) | training (`train_seq.py:227`), the scorer's network lookup, B.30 replay tables, B.37/B.64/B.69 second stage, B.42, B.49, B.51 | kept in the array; they fall in one-row blocks that `build_blocks` skips (`:282`), but they enter statistics computed straight from the grid (`run_b37.py:86-89 fast_stats`) | NaN, `_stuck_repeat_mask` per block (dt >= 0.5 s), both targets, after the cut (`:149-158`) | NaN at load if `CES_TI_SPIKE_CUT_EV` (or the argument) > 0 (`:129-130, 147-148`) |
| L-win `dataset.KSTAR_CES_Dataset` (`dataset.py:115-148`) | the scoring harness (1.4) | dropped (only rows with >= 1 observed target are kept, `:138`) | NaN, same rule (`:136-137, 181-215`) | NaN at load from env (`:82-84, 134-135`) |
| L-kfx `b56_kfx/kfx_model.py:25-43` | the filters KF0/KFX/Kc/Kg (B.56, B.57, B.60 addendum, B.61, B.66) | dropped (`iloc[1:-1]`) | **kept** | NaN (hard-coded 3000) |
| L-F `analyze_file_rules.F.load` (`:48-55`) | file rules, B.35, B.58, B.61c, B.62 X/V | dropped | **kept** | not cut by the loader; B.35 (`run_b35.py:39`), B.58, B.61c (`uc_blip_shape.py:25-26`), B.62 X/V add a hard-coded 3000 cut; the file-rules blip table has none |
| own | `fig_missing.py` (own held rule, no cut first, `iloc[1:-1]`); `kstar_level.py` (0 < T_i < 3000, no held removal, >= 20 values per discharge); `kstar_regime.py` (T_i cut, T_i held kept; V_rot held rule without the 0.5 s segment split, then abs(V) > 1000 removed) | | | |

In every network batch the cut and the held rule are applied at load, so they reach the training
targets, the T_i/V_rot history inputs, the NaN-aware target statistics, the validation loss used
for early stopping, the reference anchors (hold, causal GP, PCHIP, offline GP) and the scored rows.
`metrics.json` of a seq run does not record the cut; it is fixed only through the runner's env.
Held T_i values are rare (one bit-identical repeat among about 226,000 finite T_i), so "held kept"
is immaterial for T_i and material for V_rot.

### 1.3 Training of the networks

`train_seq.py`: target statistics and (with `CES_PER_SHOT_NORM=0`) fast-channel statistics from
the manifest's train files only, NaN-aware, after cut and held removal (`:227-228`,
`seq_data.fit_stats :175-189`); `CES_PER_SHOT_NORM=1` standardises the fast channels with the whole
discharge (`seq_data.py:250-261`). AdamW 1e-3, weight decay 1e-4, batch 16 blocks, clip 1.0, loss =
masked MSE + 0.1 relu penalty. Early stopping on validation masked MSE with patience 6 and the
best-validation checkpoint (`:288-315`) unless `CES_SEQ_FIXED_EPOCHS=1`; epoch cap
`CES_SEQ_EPOCHS` (default 30, `:198`).

| Env code | Definition |
|---|---|
| GATE | `b1_gate/run_b1_gate.py:49-66` `GATE_ENV`: W = 2, `CES_MAX_VAL_SAMPLES=40000`, `CES_MAX_TEST_SAMPLES=48000`, `CES_MAX_SAMPLES_PER_FILE=500`, `CES_DROP_STUCK_TARGETS=1`, `CES_TI_SPIKE_CUT_EV=3000` |
| E29 | `b29_causal/run_b29.py:99-133` `run_env`: shell `CES_*` stripped, GATE + data dir, population cut, per-arm `CES_PER_SHOT_NORM` (`:61-66`: 0 for `norm0`/v2, 1 for the twin v2h), MSE loss and selection, cap 30 (`:57`), batch 16, `CES_SEQ_FIXED_EPOCHS=0`, CUDA, `CES_SEED = CES_INIT_SEED` = split seed |
| E39s / E39p | `b39_primary/run_b39.py:61-83`: E29 with cap 80 (`:37`); `split_env` sets the init; `pooled_env` sets `CES_SPLIT_DIR=.b27_split_r{r}_f{k}`, `CES_SEED = 2701 + r`, `CES_INIT_SEED = 27000 + 10 r + k`, `CES_CONTROL_METRICS=.b27_ctrl_cut_r{r}_f{k}` |

Model codes: **NET80** = seq_v2 causal variant (v2, fast channels with train-file statistics),
cap 80, early stopping. **TWIN80** = history-only twin v2h, cap 80 (env says
`CES_PER_SHOT_NORM=1`, but v2h never reads the fast channels, `model_seq_v2h.py:10-12`).
**NET30** = the same causal variant, cap 30 (`.b29_norm0_cut_s{s}`). **WD30** = the
whole-discharge-standardised variant, cap 30 (`.b1_seqv2_s{s}_i{s}`, `.b27_seqv2_*`).

### 1.4 The scoring harness and its rows

`seq/eval_seq.py` scores on `evaluate.build_clean_val_subset` (`evaluate.py:95-131`):

1. L-win arrays, window W = `CES_WINDOW_SIZE` = 2, no augmentation; a sample is a kept row whose
   previous kept row lies in the same contiguous block (dt < 0.5 s) (`dataset.py:410-421`).
2. T_i is scored only if it is observed, the previous kept row has T_i observed (`has_obs`,
   `eval_seq.py:294`, `evaluate.py:55-77`) and every reference (persistence, linear, PCHIP,
   ar_local, offline GP, causal GP) is defined (`eval_seq.py:346-350`); the last two require a later
   T_i in the block.
3. **Per-run random cap**: `select_seeded_random_indices(indices, CES_MAX_VAL_SAMPLES = 40000,
   CES_SEED + 202)` (`evaluate.py:130`; `eval_seq.py:163, 199-201`; `dataset.py:547-553`, a cap
   <= 0 keeps everything).

| Family | Eligible samples per run (measured) | Cap binds? | Scored T_i rows (measured) | Max scored T_i rows of one discharge |
|---|---|---|---|---|
| pooled folds (r2, r5, r9 checked) | 40,666-52,744 | yes, every fold: `eval_samples` = 40000 | 191,934-192,145 per re-partition (about 86 % of eligible samples) | 1,453 (r2); never below 685 per fold |
| four splits | 33,814 / 36,823 / 37,229 / 35,393 | no | 32,589 / 35,582 / 35,759 / 34,038 | 870 (split 42) |
| time split | 21,142 | no | 20,062 | |

There is **no per-discharge cap** in the scoring path: `CES_MAX_SAMPLES_PER_FILE=500` (GATE) is
read only by `ces_prediction/train.py:479` (training of the window family). All cut runs of a pooled
fold draw the same subset (same seed, same eligible list), and the analyses verify it
(`analyze_b39.same_rows`); the inclusive population draws a different subset (B.65).
Stored per run as `comparison_errors_test.npz`; its `CES_TI_row` is the index in the L-win file
array, mapped to grid rows by `run_b45.compact_index`.

### 1.5 Other row sets

| Code | Definition | T_i rows (s42 / s1 / s7 / s123) |
|---|---|---|
| R-ES | 1.4 on the four splits or the time split (cap not binding: every eligible row) | 32,589 / 35,582 / 35,759 / 34,038 |
| R-ES40k | 1.4 on a pooled fold (random 40,000-sample subset) | 191,934-192,145 per re-partition |
| R-B54 | R-ES40k of r2-r9 mapped to grid rows (`run_b54.py:112-114, 180-204`) | as R-ES40k |
| T30 | B.30 replay table `data/.b30/rows_cut_s{s}.npz` (`b30_thinned/run_b30.py:101-209`): L-seq grid, every observed row with >= 1 earlier visible observation in its seq block, no cap; columns bb (WD30), bb80, hist80 (TWIN80), causal (NET30); its own causal GP | |
| T48 | T30 with the causal column replaced by the B.39 network `.b39_v2_cut_s{s}_i{s}` (NET80) (`b48_reread/run_b48.py:103-137`) | |
| P41 | T30/T48 at k = 1 with hold, GP, hist80, causal, y finite (`run_b41.py:115-116`); a superset of R-ES (+332 to +437 rows: rows whose previous kept row lacks T_i or that have no later value) | 32,921 / 35,936 / 36,196 / 34,428 |
| T30a | the B.30 analysis mask: P41 and `has_future` and every estimator incl. PCHIP finite (`run_b30.py:312`) | 32,818 / 35,831 / 36,094 / 34,327 |
| R-B34 | P41-type rows with a finite 10-row window feature (drops about the first 10 rows of each segment) | 32,127 / 35,138 / 35,357 / 33,662 |
| R-fit64 | B.64 second-stage fit rows (`stage2_b64.py:227-240`): every observed T_i with an earlier T_i in its block, natural cadence, no cap, on inner held-out + outer validation discharges | r2 f0: 137,227 + 44,844 (network arm), 140,394 + 45,911 (twin arm) |
| TRI | B.57 triple rows: t-1, t, t+1 all in R-ES40k, consecutive grid rows, stored dt < 15 ms (`run_b57.py:169-177`); depends on which rows survived the cap | |

Two causal-GP implementations coexist: the scorer's (stored in the npz; used by B.39, B.45, B.49,
B.54 and everything on R-ES40k) and the replay machinery's (T30/T48: B.41, B.42, B.44, B.48, B.50,
B.51, B.52). They differ on 16 rows of split 42 and 15 of split 7 (PK 5, trap 7).

### 1.6 "Rows after an absent frame": the code definitions

| Where | Definition |
|---|---|
| file rules (`analyze_file_rules.py`) | step 15-45 ms with six contiguous rows before |
| B.41, B.48 blip part (`run_b41.py:438-447`) | rows 1-15 after a 20-40 ms step, 5 regular steps and >= 3 T_i before |
| B.54, B.60, B.61c (`run_b54.py:118-137`) | rows 1-3 after a 15-45 ms step, 5 regular rows and >= 3 finite T_i before |
| B.55 (`run_b55.py:314-326`) | rows 1-3 after the step, no condition before |
| B.56 KX6, B.57 quiet mask (`run_b56.py:365-378`) | the 10 rows after a step of 15 ms to 0.5 s |
| B.45 classes (`run_b45.py`), B.51 quiet rows | 10 rows after an absent frame; class 7 = first 10 rows of a segment, assigned first |
| B.30 sensitivity (`sensitivity_b30.py`) | a 100 ms window after a 15-45 ms step |

---

## 2. The ledger

Columns as requested: (1) pool, (2) split, (3) T_i outlier rule, (4) held values / V_rot,
(5) scored rows and whether every estimator is on the same rows, (6) normalisation,
(7) model / epoch cap / stopping, (8) initialisations, (9) output. Codes are those of section 1.
"C-all" = 3 keV cut at load with all application points of 1.2; "H-rm" = held removed at load,
both targets. V_rot is never scored in these batches; it enters only as an input channel
(held removed) and shapes R-ES (a row with only V_rot is a kept row, so the next T_i is not scored).

### A. Results the main text uses (pooled, filters, second stage, and the four-split replays it cites)

| Result | (1) | (2) | (3) | (4) | (5) | (6) | (7) | (8) | (9) |
|---|---|---|---|---|---|---|---|---|---|
| B.39 arm P, 4.32 (main table a, network rows) | U639 | pooled r2-r9 (40 folds) | C-all | H-rm | R-ES40k; network, twin and references on identical rows (`analyze_b39.same_rows`) | NET: 0; twin: nominal 1, unused | NET80 / TWIN80 (E39p); epochs run 18-56 (v2), 19-80 (v2h); one run at cap (`.b39p_v2h_cut_r3_f4`) | 1 per fold | `docs/paper/b39_primary.{json,md}` (`analyze_b39.py`) |
| B.54, 4.33 (Table where) | U639; per-discharge readouts need >= 50 rows | r2-r9 | C-all; extra thresholds only to label strata of scored rows (y and hold in [150, 2500] eV, abs(y - hold) <= 1 keV, not within 30 ms after a low-BES row; `run_b41.py:108-112`) | H-rm; propensity target "unusable" = missing, held or > 3 keV | R-B54, stored errors, no network run | as B.39 P | as B.39 P | 1 | `b54_pooled_rows.{json,md}` |
| B.55, 4.34 (Table carry, block 1) | parts G, D: scored discharges of U639; part R: RAW12 rows inside the raw windows | r2-r9 | C-all (checked, `run_b55.py:92-111`) | H-rm | R-B54; the 40 v2 networks re-run on CPU with shifted inputs (median 0.026 eV from the GPU errors); twin and references are stored GPU errors | 0 | NET80 | 1 | `b55_sample_timing.{json,md}` |
| B.57, 4.38 (floor, removable error, persistence) | U639 | r2-r9 (filter rows include r2, its development set) | C-all (networks); filters see L-kfx | H-rm (networks); filters keep held | MSE quantities on R-B54; floor and persistence on TRI; signed errors from B.55, the twin re-run on CPU, B.56 | 0 | NET80, TWIN80, filters | 1 | `b57_yardsticks.{json,md}` |
| B.60, 4.36 (share within 10 % / 20 %) | U639 | r2-r9; addendum (filters) r3-r9, r2 apart | C-all | H-rm | R-B54; 50 eV cells are row strata only | 0 | NET80; filters | 1 | `b60_relative_error.{json,md}` |
| B.61 A/B, 4.40 (Huber, 1 % trim, discharge removal; main table a) | U639; part B removes the top-k discharges | r2-r9; filter rows r3-r9 | C-all; Huber threshold 200 eV; trim = rows at or above the 99th percentile of hold's squared error within each repeat (`run_b61.py:43-46, 67`) | H-rm | R-B54 + B.56 filter columns | 0 | NET80 | 1 | `b61_absolute_robust.{json,md}` |
| B.66, 4.43 (Table where, burst classes) | U639; rate classes from 611 discharges with a defined rate (28 short files out) | r2-r9; filter r3-r9 | C-all; 10 ms BES detector on the cut, held-free grid | H-rm | R-B54 | 0 | NET80 | 1 | `b66_regimes.{json,md}` |
| B.56 KFX/KF0, 4.37 (main table, carry block 4) | fit set: F641 minus the fold's discharges with scored rows (= train + val + s30967, s31702 + test discharges without rows) (`run_b56.py:201-203`) | spec on r2; confirmation r3-r9; also four splits and the time split | L-kfx: T_i > 3000 -> NaN | **held kept** in fit loss and filter state (`kfx_model.py:33-43, 102-106`) | stored R-B54 rows located by value match (3e-4 eV), every row checked found, so filter and network share rows (`run_b56.py:126-154, 296-323`) | none: raw log BES, common mode and radial difference, change clipped at 0.2 | 3- or 5-parameter filter, sequential fit, resets at gaps >= 0.5 s | none | `b56_kfx.{json,md}` |
| B.59, 4.44 (Table carry, block 2) | U639 | **r2-r4** (15 folds) | C-all | H-rm | R-ES40k, identical to B.39 rows (`analyze_b59.py:100`); blip rows from B.54 | 0 | 4 arms x E39p, cap 80, early stopping; arm L shifts standardised fast channels one row in training and scoring (`fast_shift.py:46-68`); partners B.39 P | 1 | `b59_inputs.{json,md}` |
| B.64, 4.45 (paper's estimator; abstract, main table a and c) | U639 | **r2-r4** | C-all | H-rm | test: R-B54 rows of r2-r4; second stage fitted on R-fit64 (inner held-out discharges of 4 inner folds + the outer validation discharges); applied where all 15 network inputs exist, otherwise first stage kept in both arms (2.35 % of rows, `stage2_b64.py:352-363`); twin arm fitted on more rows than it is applied to | network 0; second-stage window features standardised over all non-test files incl. validation, the two files outside U639 and edge rows (`run_b37.py:86-89`) | 120 inner NET80/TWIN80 runs (E39p with inner split; 4 at cap) + B.39 outer networks; HistGBR on the residual | 1 per inner and outer fold | `b64_two_stage.{json,md}` |
| B.53, 4.39 (Table carry, block 3) | S638 | four splits | C-all | H-rm | R-ES (cap not binding); all networks on the twin's rows (`analyze_b53.py:95`) | arm C: twin env (1, unused); extra BES-component channels standardised with training-discharge constants, edge rows and non-positive BES rows set to 0 (`make_features.py:92-127`); arm F: 0 | arm C: v2hx, cap 80, early stopping; **arm F: fixed 40 epochs, final weights**; partners `.b39_v2_cut_s{s}_i{s}`, `.b29a1_v2h_cut_s{s}` (cap 80, early stopping) | 1 (= split seed) | `b53_components.{json,md}` |
| B.48, 4.28 (Fig. use b: delay; thinning in supp) | S638 test discharges (96 per split) | four splits | C-all | H-rm | T48 / P41; delay readout on B.42's own rows per delay L (below); k = 5 cell adds `has_future` and bb, bb80 finite | NET80 0; twin 1 unused | NET80 (`.b39_v2_cut_s{s}_i{s}`) vs TWIN80; on s1 and s123 the NET80 weights are byte-identical to NET30 (measured, sha256) | 1 | `b48_reread.{json,md}` (fig_paper.py reads `["b42"]`) |
| B.65, 4.46 (not yet in the manuscript; main text still cites the older whole-discharge inclusive readout) | U639 | **r2-r4** | **C0**: values > 3 keV kept in training targets, history inputs, target statistics, early stopping, references and scored rows (`run_b65.py:44, 55-63`; references re-scored by `refs_b65.py`) | H-rm (applied on the uncut values) | R-ES40k drawn from the inclusive eligible list: a different random subset from the cut rows (r2 f0: 37,612 vs 37,483 T_i rows). The "<= 3 keV" readout is the mask `y <= 3000` on these rows (`analyze_b65.py:253`): inclusive-trained networks, histories and references still contain > 3 keV values, 1 % trim recomputed in the cell | 0 | NET80/TWIN80 under C0, control stats `.b27_ctrl_incl_r{r}_f{k}` | 1 | `b65_inclusive.{json,md}` |
| B.67, 4.47 (not yet in the manuscript) | U639 | **r2-r4** | C-all | H-rm | R-ES40k (identical to B.39); row classes from B.54 and B.66 | 0 | v2s / v2a, E39p, cap 80 (one run at best epoch 78), early stopping; partners B.39 P | 1 | `b67_gated.{json,md}` |
| B.68 (registered, finished, untracked) | U639 | **r2-r4** | C-all; the auxiliary target (offline GP on harness arrays) is built under the cut | H-rm | R-ES40k, identical to B.39 | 0 | v2 + auxiliary head (weight 0.5), E39p, early stopping on the main loss only | 1 | `docs/paper/b68_auxtarget.{json,md}` (untracked) |
| B.69 (registered, partly run) | TS: test 97 (95 with rows), train 413 | time split; the four seeds differ only in initialisation and inner folds | C-all | H-rm | R-ES (cap not binding: 21,142 samples, 20,062 T_i rows, identical across seeds) | 0 | E39p with the time-split dir, control stats `.b5_camp_cut_win_s{s}`, cap 80, early stopping; second stage as B.64 (`stage2_b69.py:86-99`) | 4 (42, 1, 7, 123) | not yet written. Status: outer 8/8 scored; inner 17 of 32 weights; no `data/.b69/rows_*`, no stage-2 file |

### B. Four-split ledger (supplement)

| Result | (1) | (2) | (3) | (4) | (5) | (6) | (7) | (8) | (9) |
|---|---|---|---|---|---|---|---|---|---|
| B.39 arm A, 4.26 (+ ablations v2e, v2b, v2n) | S638 | four splits | C-all | H-rm | R-ES, identical across all runs of a split | 0 (twin 1, unused) | NET80, TWIN80 (E39s); `.b39_v2b_cut_s42_i42` reached the cap; two twins stopped at 11-12 epochs | 3 per split (split seed + next two); ablations 1 | `b39_primary.{json,md}`, `b39_inits.md` |
| B.49, 4.27 | S638 | four splits | C-all | H-rm | R-ES (CPU re-run of the six networks, each checked against its stored error) | 0 | NET80 x 3, TWIN80 x 3 | average of 3 | `b49_average.{json,md}` |
| B.45, 4.24 | S638 | four splits | C-all | H-rm | R-ES with row classes on the cut grid; class 7 (first 10 rows of a segment, 752-825 rows) assigned before classes 1-2 | 0 | ablation nets cap 80 | 1 | `b45_ablation_rows.md` |
| B.50, 4.29 | S638 | four splits | C-all; propensity target includes held and > 3 keV as "unusable" | H-rm | T48, k = 1, causal/hist80/GP/hold/y finite and position >= 10 in a segment: R-B34 counts | 0 | NET80 | 1 | `b50_missing_like.md` |
| B.51 part B, 4.30 | S638 | four splits | C-all | H-rm | T48 k = 1 with all second-stage features finite (89-93 of 96 discharges); BES-network comparison on the intersection with R-ES | twin 1 (unused); features from training files | TWIN80 + HistGBR fitted on validation files | 1 | `b51_bes_modes.md` |
| B.52, 4.31 | S638; D needs >= 50 rows (85/88/93/91 discharges) | four splits | C-all; V: y <= T **and hold <= T** on scored rows only (`run_b52.py:414`) | H-rm | P41 on T48; E rows on R-ES; L: >= 3 earlier visible T_i and 10 earlier fast rows | 0 | NET80; ridge filters LH/LF fitted on training files | 1 (E: 3) | `b52_grading_readouts.{json,md}` |
| B.41, 4.20 | S638; U1 traces RAW12 (7 test discharges for estimator traces) | four splits | C-all; value rules label rows only | H-rm | P41 on T30 | causal 0 | **NET30 vs TWIN80** (caps differ) | 1 | `b41_events.md` |
| B.44, 4.22 | S638, thirds by discharge median T_i | four splits | C-all | H-rm | P41 on T30 | 0 | **NET30 vs TWIN80** | 1 | `b44_strata.md` |
| B.42, 4.23 | S638 (94-96 discharges at L >= 1) | four splits | C-all | H-rm | own rows per delay L: T_i measured with >= 1 observation at or before r-1-L (`run_b42.py:145-162`); at L = 0 equal to P41; rows shrink with L (s42: 32,921 / 32,830 / 32,735 / 32,555); the RMSE ratio to L = 0 compares different row sets | 0 | **NET30** and TWIN80 run on CPU with rebuilt history channels | 1 | `b42_delay.md` |

### C. Model-free readouts the paper cites

| Result | (1) | (3) | (4) | Rows / events | (9) |
|---|---|---|---|---|---|
| File rules, PK 2 (Data section, Fig. rows, blip table) | F641 (asserted) | **none** | **kept**; "fresh V_rot" = differs from the previous row | absent frame per 1.6; raw blip on RAW12; blip table T_i row 1 rests on 1,215 of the 1,394 events | `file_rules.{json,md}` |
| Missingness, PK 2 | F641, `iloc[1:-1]` | none before the held rule | own rule; V_rot gaps exclude held stretches | runs of empty cells | `ppcf/fig_missing.json` |
| Level, PK 2 (528 eV) | F641 -> 603 with >= 20 values | 0 < T_i < 3000 | not removed | all rows | `ppcf/kstar_level.json` |
| Regime number, PK 2 | F641 -> 601 (T_i) / 382 (V_rot) | 3000 cut | T_i: kept; V_rot: own rule without segment split, abs(V) > 1000 removed | triples 10 ms apart, >= 8 pairs | `ppcf/kstar_regime.json` |
| B.35, 4.17 | F641 -> 1,366 events, 498 discharges | hard-coded 3000 | **kept** | absent step, five regular steps and >= 3 T_i before | `b35_blip_response.md` |
| B.40, 4.21 | F641 | cut grid (and an inclusive grid) | H-rm | triples at 10 +- 0.1 ms; row-table part on T30 k = 1 | `b40_noise.md` |
| B.43, 4.21b | F641 | S, H: cut; F: inclusive | S, H: H-rm; F: raw | 200 ms blocks; part H discharge 31923 only | `b43_target_checks.md` |
| B.46, 4.25 | F641 | cut | H-rm | isolated absent frame (none within 400 ms), >= 10 T_i per window; controls every 0.5 s from 0.6 s after a segment start | `b46_beam_off.md` |
| B.51 part A, 4.30 | F641 | cut (association); hard-coded cut (control) | H-rm (association); kept (control) | 20-row blocks with >= 10 T_i | `b51_bes_modes.md` |
| B.58, 4.35 (Fig. blips, Table blips) | F641 -> 1,262 events, 489 discharges (T_i ratios use fewer; n not written) | hard-coded 3000 | **kept** (registration says "cut population") | 1.6 file-rule events | `b58_blip_calibration.{json,md}` |
| B.61c, 4.40 (blip dose) | F641 minus 37 files with < 30 rows | hard-coded 3000 | **kept** | 15-45 ms steps, 6 regular rows and >= 3 T_i before | `b61_blip_dose.{json,md}` |
| B.62, 4.41 | E: RAW12 (31 events, 10 discharges); X, V: F641 events | X, V: hard-coded 3000 | V: previous-row "fresh" rule | per part | `b62_beam_return.md` |
| B.63, 4.42 | RAW12 (36 events) | no T_i | n/a | 240/240 pseudo-event halves | `b63_blip_magnetics.md` |

### D. Older network results the supplement still cites

| Result | (1) | (2) | (3) | (4) | (5) | (6) | (7) | (8) | (9) |
|---|---|---|---|---|---|---|---|---|---|
| B.27c, 4.1 (supp S3) | U639; scored 633-635 per repeat | pooled r2-r9 (r0-r9 secondary) | C-all and C0 | H-rm | R-ES40k (cap binds on all 100 fold runs) | **1 (whole discharge)** | WD30; **cap 30 reached in 26 of 40 cut runs** (measured), 1/40 inclusive | 1 | `data/.b27c_hier.json` (copied to `docs/paper/b27c_hier.json`; stores the bootstrap median) |
| B.29 and addendum 1, 4.2-4.3 | S638 (cut), I639 (incl), TS | four splits + time split | C-all / C0 | H-rm | R-ES; `paired_model_compare` asserts identical rows | 0 (norm0), 1 (v2h, backbone), expanding (arm C) | cap 30; addendum cap 80. The cap-80 backbone `.b29a1_v2_cut_s{s}` stopped at 25/20/20 epochs and equals the frozen backbone on splits 42, 1, 7; time-split inclusive norm0 runs had best epochs 1-5 | 1 (time split: 4) | `b29_causal.md`, `b29_addendum1.md` |
| B.30 (+ sensitivity, stack, offline stack), 4.4, 4.5, 4.9 | S638 / I639 test discharges | four splits | C-all / C0 | H-rm | T30a (k = 1 to 10 thinning); sensitivity filters y **and** hold (< 150 eV, > 2.5 keV, abs(y - hold) > 1 keV); stack weights fitted on half of the test discharges | WD (bb, bb80), 0 (causal) | WD30, WD80 (bb80), NET30, TWIN80/TWIN30; the PK 4.4 "network vs twin" row is bb80 vs hist80 while the other "network" rows are bb; posthoc increment = NET30 vs TWIN80 | 1 | `b30_thinned.md`, `b30_sensitivity.md`, `b30_stack*.md` |
| B.31, 4.6 | RAW12 test discharges (31921 excluded from the rules) | four splits | C-all | H-rm; "T_i missing" includes held and > 3 keV | T30a-type, `has_future` required (not in the registration), rows within 30 ms after a blip and 200 ms of the raw start dropped | WD | bb80 vs hist80 | 1 | `b31_bursts.md` |
| B.32, 4.14 | F641 (part A, 1,203 rows, 449 discharges); RAW12 (part B) | four splits (model part) | hard-coded 3000 | **kept** (part A, posthoc) | model part: T30 k = 1 joined to events | 0 | **NET30 vs TWIN80** | 1 | `b32_*.md`, `fast_departure.md` |
| B.34, B.36, B.37, B.38, 4.16, 4.18, 4.19 | S638 (I639 for B.37 inclusive) | four splits | C-all | H-rm; B.37 F "missing" includes held | R-B34 (shared by all four); B.37 V fitted on the split's validation files (the network's early-stopping set); B.38 on out-of-fold predictions of 16 fold networks | features z-scored over all non-test files incl. edge rows | NET30; B.36/B.37 Q pair NET30 with TWIN80; B.38 fold nets `.b38_norm0_cut_s{s}_f{k}` cap 30 | 1 | `b34_*.md` ... `b38_stage2.md` |

---

## 3. Groups of identical treatment

| Group | Members | Shared definition |
|---|---|---|
| G1 | B.39 P, B.54, B.55 (G, D), B.57 (MSE parts), B.60, B.61 A/B, B.66 | U639, pooled r2-r9, C-all, H-rm, R-ES40k / R-B54, NET80 + TWIN80 (E39p), 1 init per fold |
| G2 | B.59, B.67, B.68 | G1 restricted to r2-r4, with arm-specific networks under the same E39p |
| G3 | B.64 | G2 + a second stage fitted on R-fit64 with window features standardised over all non-test files |
| G4 | B.65 | G2 under C0 (a different 40,000-row subset) |
| G5 | B.56, B.60 addendum, B.61 filter rows, B.57 filter parts | G1 rows; filters on L-kfx (held kept), fitted on all non-test files; specification on r2, readouts on r3-r9 (r2-r9 for the floor and persistence) |
| G6 | B.39 A, B.49, B.45, B.53 C | S638, four splits, C-all, H-rm, R-ES (no cap), NET80/TWIN80 (E39s); 3 inits (B.39 A), average of 3 (B.49), 1 (B.45, B.53 C) |
| G6' | B.53 F | G6 with fixed 40 epochs |
| G7 | B.48, B.50, B.51 B, B.52 | four splits, T48 / P41 (or R-B34-type subsets), NET80 vs TWIN80, replay GP |
| G8 | B.41, B.44, B.42, B.30 posthoc, B.32 model part, B.34-B.38 | four splits, T30 / P41 / R-B34 / own delay rows, NET30 (vs TWIN80 where a twin is used), replay GP |
| G9 | B.69 | TS, R-ES, NET80 x 4 inits + second stage |
| G10 | B.29 time split, B.56 time split | TS, R-ES, NET30 x 4 inits (B.56: network side `.b29_norm0_camp_cut_s{s}`) |
| G11 | B.55 R | RAW12 rows of the pooled networks |
| G12 | B.27c, B.29 (four splits), B.30 bb rows, B.31 | whole-discharge standardisation (WD30 / WD80) |
| M1-M4 | model-free | M1 cut + held removed (B.40, B.43 S/H, B.46, B.51 A); M2 cut + held kept (B.35, B.58, B.61c, B.62 X/V, B.32 A); M3 no cut + held kept (file-rules blip table); M4 own rules (fig_missing, kstar_level, kstar_regime); plus RAW12 (B.62 E, B.63) |

Counting each network group and each model-free rule set once: 17 distinct treatments
(13 network-side: G1-G12 with G6' separate; 4 model-free: M1-M4, the RAW12 readouts falling
under their rules). The main text uses seven network-side treatments (G1, G2 via B.59, G3, G5,
G6 and G6' via B.53, G7 via the B.48 delay replay in Fig. use b) and model-free numbers from all
four rule sets (data section, beam-blip figure and table).

---

## 4. Differences that matter

Variables that are not the same across the results the main text uses, and what it would cost to
put them on one definition. "CPU" = re-reading stored errors, or re-running stored networks
forward on a CPU; "GPU" = retraining.

### 4.1 Re-partition / split set

| Value | Main-text results |
|---|---|
| r2-r9 | G1 (network rows of the main table, Table where, Table carry block 1, floor) |
| r2-r4 | B.64 (the paper's estimator), B.59 (Table carry block 2); B.65, B.67, B.68 (not yet in the text) |
| r3-r9 | filter rows (main table a, b; Table carry block 4) |
| r2 | filter specification; included in the filter's floor and persistence rows |
| four splits | B.53 (Table carry block 3), B.48 delay replay (Fig. use b) |

- To r2-r9 for everything: GPU for B.59 (4 arms x 25 folds = 100 runs), B.64 (8 inner runs x 25
  folds = 200 runs, then CPU second stage), B.65 (50), B.67 (50), B.68 (25), B.53 on pooled folds
  (5 networks x 40 folds = 200). The filter on r2: CPU (already computed; only the inclusion rule
  changes). The delay and thinning replay on the 80 stored pooled networks: CPU (forward runs as in
  B.55, with the B.42/B.30 history rebuild).
- To r2-r4 for everything: CPU only (restrict the stored errors of G1 and G5 to r2-r4).

### 4.2 Sampling of scored rows

| Value | Results |
|---|---|
| random 40,000 samples per fold (about 86 % of eligible rows, any number per discharge) | every pooled result: G1-G5, B.27c |
| every eligible row | four splits (G6-G8), time split (G9, G10) |

The manuscript describes a per-discharge cap of 500 that the code does not have (section 5, item 1).
TRI (the floor) and the "never-scored" discharge count depend on which rows survived the cap.

- To score every eligible row in the pooled results: CPU. Re-run `eval_seq.py` with
  `CES_MAX_VAL_SAMPLES=0` and `CES_SEQ_DEVICE=cpu` on the stored weights (B.39 P 80 runs, B.59 60,
  B.65 30 plus its 30 reference runs, B.67 30, B.68 15; B.27c 40 for the supplement), then re-run
  the CPU analyses (B.39 analyze, B.54, B.55, B.57, B.60, B.61, B.66, the B.56 row matching, the
  B.64 second-stage application). No retraining.
- To impose a true per-discharge cap instead: CPU, the same re-scoring plus a subset rule.

### 4.3 Row definition

| Value | Results |
|---|---|
| R-ES / R-ES40k (previous kept row carries T_i; a later T_i exists) | G1-G6, G9 |
| P41 / T48 (any earlier T_i in the block; no later value needed) | B.48 (Fig. use b), B.50-B.52, older B.41/B.44 |
| own per-delay rows (shrinking with L) | B.42 / B.48 delay readout |
| R-fit64 (second-stage fit) | B.64 fit only |
| TRI | B.57 floor |

- Restricting P41/T48 readouts to R-ES rows: CPU (R-ES is a subset of P41; the B.52 join is
  100 %). The delay readout on a fixed row set across L: CPU.

### 4.4 Held values

Networks, references and all G1-G9 rows: removed. Filters (G5) and the model-free blip readouts
(B.35, B.58, B.61c, B.62 X/V, file rules): kept. Immaterial for T_i (one held repeat), material for
any V_rot number (file rules "fresh V_rot", B.62 V, kstar_regime). Aligning: CPU (refit the
filters on L-seq-style held-free arrays; re-run the model-free scripts with `_stuck_repeat_mask`).

### 4.5 T_i outlier rule

All main-text network results: 3 keV cut at load with every application point of 1.2. Different:
B.65 (C0 everywhere, and a "<= 3 keV" mask that is not the cut population), the file-rules blip
table (no cut), `kstar_level` (0 < T_i < 3000), the hard-coded cuts of M2 (same value, applied
without held removal). Value thresholds on scored rows only: B.54 plain rows (150-2500 eV,
abs(y - hold) <= 1 keV, low BES), B.52 V (y and hold <= T), B.60 50 eV cells. A B.65 readout on the
cut population: CPU (score the 30 inclusive networks with the cut harness; the cut references
exist). Any change of the training cut: GPU.

### 4.6 Epoch cap, stopping, and the network that is read

| Value | Results |
|---|---|
| cap 80, early stopping (patience 6, best-validation checkpoint) | G1-G4, G6, G7, G9 |
| fixed 40 epochs, final weights | B.53 F |
| cap 30 | B.41, B.42, B.44 and every G8 result; G10 (time split networks); B.27c (cap reached in 26/40 cut runs) |

On splits 1 and 123 the cap-80 network of B.48 is byte-identical to the cap-30 network, so B.48
and B.41/B.44 coincide there by construction. Aligning anything on cap 30 to cap 80: GPU (the
time-split networks: B.69 already retrains them at cap 80).

### 4.7 Initialisations

1 per fold (all pooled results, B.64 inner and outer), 3 per split (B.39 A), average of 3 (B.49),
1 (ablations, B.53, B.48 replay), 4 (time split). More initialisations on the pooled folds: GPU
(2 more per fold for B.39 P = 160 runs; the same factor for each r2-r4 batch).

### 4.8 Normalisation and fitting sets

Network fast channels: training-discharge statistics everywhere in the main text (the twin's
nominal `CES_PER_SHOT_NORM=1` is unused). Filters: raw log BES, no standardisation, fitted on all
non-test discharges (train + validation + the two files outside U639). Second stage (B.64, B.69):
features standardised over all non-test files including validation, the two outside files and edge
rows; fitted on inner held-out plus outer validation discharges (the outer network early-stopped
on the latter). Train fraction differs: 383-384 train discharges (pooled) against 414-415 (four
splits) and 413 (time split). Aligning the second-stage or filter statistics: CPU. Aligning train
fractions: GPU.

### 4.9 Discharge pool

"639 discharges" (PK 2, main text) is the universe U639; the cut four splits use 638; pooled
scoring covers 633-635 per re-partition; B.66 classes 611; B.61c 604; kstar_level 603; filters
are fitted on F641-based sets. Only wording or a CPU re-read is needed, except where a training set
would change (GPU).

### 4.10 Event and stratum definitions

Seven definitions of "after an absent frame" (1.6); two causal-GP implementations (1.5). Aligning:
CPU.

---

## 5. Where code and prose disagree

1. **Per-discharge cap of 500.** `main_ppcf.tex:387-391` and `supp_ppcf.tex:1034-1037` say the
   scored rows of a test discharge are "a seeded subset of at most 500". The scorer has a per-run
   cap of 40,000 samples (`evaluate.py:130`, `eval_seq.py:163`); up to 1,453 T_i rows of one
   discharge are scored in a pooled fold, 870 on split 42; `CES_MAX_SAMPLES_PER_FILE=500` is read
   only by `train.py:479`. The 15th grader's point "the place of the 500-row cap" (PK 4.13) rests
   on this text.
2. **639 discharges.** PK 2 ("분할에 쓰는 방전은 639개") and PK 4.1 / `PREREGISTRATION_B27.md` §7
   ("scores all 639 shots"): the cut four splits use 638; pooled scoring reaches 633-635 per
   re-partition.
3. **B.27c (PK 4.1).** The cap of 30 is reached in 26 of 40 cut runs; the json stores the
   bootstrap median, which PK 4.1 quotes as the point value.
4. **B.65 (PK 4.46).** "같은 run을 3 keV 이하 행에서" is a mask on the inclusive rows with
   inclusive histories and references, not the cut population; `PREREGISTRATION_B65.md` says
   "every measured row of the test discharges", the code scores the 40,000-sample subset.
5. **Held values "removed everywhere"** (`main_ppcf.tex:224-225`, `supp_ppcf.tex:228-230`, PAPER_DESIGN 3.2):
   the filters and the model-free blip readouts keep them (immaterial for T_i).
6. **B.52 V (PK 4.31)** is described as limiting scored rows to <= 2,000 eV; the code also requires
   hold <= 2,000 eV.
7. **B.42 (PK 4.23) "같은 행에서"** holds within one delay only; the 6-8 % RMSE rise compares row
   sets about 0.6 % apart.
8. **Twin pairings with two caps.** The increments quoted from B.30 posthoc, B.32 (PK 4.14
   +0.494...+0.677), B.36 and B.37 Q pair NET30 with TWIN80; PK 4.4's "network vs twin" row is
   bb80 vs hist80 while the other "network" rows are bb; the B.29 cap-80 backbone equals the frozen
   cap-30 backbone on three of four splits.
9. **B.37 F / PK 4.19** "missing or > 3 keV" also counts held values as unusable (so does the
   propensity model of B.50/B.54).
10. **B.58** registration says "cut population"; held values are not removed. **B.61c**
    registration says all 641 files; 37 short files are skipped. **B.66** registration names the
    raw-stream detector of B.31; the code uses the 10 ms BES rule (declared in its CHOICES).
11. **B.69** registration §2/§7 say the seeds differ in partition; the time split is the same for
    all seeds (corrected in its addendum 1). **B.64** registration names `split_env`; the code
    uses `pooled_env` (the intended one) and does not state that fit rows and scored rows differ.
12. **File rules blip table** (supplement): uncut population with held values kept; the T_i
    ratio of the first row rests on 1,215 of the 1,394 events.
13. **B.29 time split, inclusive causal variant**: best epochs 1-5; not stated in the records.
