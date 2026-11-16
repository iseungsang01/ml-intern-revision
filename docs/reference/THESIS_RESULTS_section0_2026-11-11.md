> 참고용 보관본. `THESIS_RESULTS.md`의 §0(2026-11-11 작성)을 2026-11-16에 그대로 옮긴 것이다. 이후 결과(B.29–B.31, MAST, 파일 규칙, 결측 수치 정정)가 반영되어 있지 않고 "Closed axes" 같은 문장은 더 이상 다음 단계의 근거로 쓰지 않는다. 지금 쓰는 기록은 `PROJECT_KNOWLEDGE.md`이다.

## 0. Current state (2026-11-11) — the two claims and where each number lives

**Protocol (§8v, `experiments/PREREGISTRATION_W2.md`).** W = 2 · held-free training and scoring ·
per-file cap 500 · file-level splits, four held-out TEST splits (seeds 42 / 1 / 7 / 123) · two
co-primary populations: *cut* (`CES_TI` > 3 keV treated as missing, 0.53 % of `T_i` rows) and
*inclusive* (no cut) · shot-clustered paired bootstrap, 10,000 resamples · every batch's hypotheses,
decision rules and numerical predictions committed to the repository before it runs.

**Main model.** `seq_v2` (`experiments/seq/model_seq_v2.py`): full-grid two-branch causal LSTM,
357,570 parameters; the `V_rot` branch reads no fast channel by construction (§8t, §8x). The
windowed GRU + attention (`model_iter009.py`) is only the W = 2 paired control.

**Claim 1 — ion temperature: a causal nowcast with lower error than future-using interpolation,
and the equation it computes.**
- Four-split skill vs PCHIP: +0.17 … +0.26 (cut), +0.23 … +0.32 (inclusive), 95 % interval
  excluding 0 on 8/8 split–population cells; vs causal GP +0.08 … +0.17, 8/8; vs offline GP
  −0.05 … +0.11, interval including 0 on 7/8 (§8ab).
- vs linear interpolation (per split, read 2026-11-11 from `bootstrap_summary.json` in
  `data/.b1_seqv2_s{s}_i{s}` and `data/.b5i_seqv2_s{s}`): cut +0.136 / +0.215 / +0.215 / +0.226,
  inclusive +0.209 / +0.210 / +0.270 / +0.288 (seeds 42 / 1 / 7 / 123), interval excluding 0 on
  4/4 in both populations. Linear is the stronger offline interpolant here (Δt ≤ 15 ms for ≥ 96 %
  of rows); the skill vs linear is 0.01–0.05 below the skill vs PCHIP.
- Shot-level repeated 5-fold replicate, overlap-corrected interval [+0.19, +0.25] (cut) /
  [+0.19, +0.31] (inclusive): 100 runs / 50 folds, §8by secondary (ii) ([+0.186, +0.253] cut;
  inclusive [+0.188, +0.307] in `docs/paper/b27_kfold_interval_R10.json`); the first 20-run pass is
  §8bw ([+0.194, +0.253] / [+0.193, +0.306]).
- Two-level bootstrap over eight new partitions (B.27c, cut): vs PCHIP +0.221 [+0.188, +0.248],
  vs persistence +0.391 [+0.369, +0.408], vs causal GP +0.113 [+0.085, +0.132], vs offline GP
  −0.010 [−0.043, +0.014] (includes 0) (§8by).
- Stress tests: reweighted onto the genuinely missing rows, skill vs persistence +0.28 … +0.44
  (8/8; values in `docs/paper/paper_numbers.json` → `mnar`, pass counts in §8ab); vs PCHIP excludes 0
  on 2/4 cut and 4/4 inclusive splits; across a campaign split +0.17 … +0.20 vs PCHIP
  (4/4 initialisations, both populations) (§8ab).
- Where the margin comes from: in the cut population, zeroing BES/ECEI/MC in the windowed control
  drops it below PCHIP (paired −0.25 … −0.43, 4/4 significant); in the inclusive population a
  history-only model keeps +0.15 … +0.23 over PCHIP (spike robustness) (§8ab). Energy equation:
  r(BES, dT_i/dt) = +0.070, r(ECEI, dT_i/dt) = +0.078 (§8ar).
- What the network computes (B.28, §8bz–§8bz-4): a 20-parameter physical filter — observation
  filter with gap-dependent gain (K₀ ≈ 0.28, K_∞ ≈ 0.51, s_K ≈ 42 ms) plus a linear response to each
  fast channel's deviation from its preceding 100 ms baseline, switched off beyond θ ≈ 3.0σ
  (Σa_BES ≈ +50, Σa_ECEI ≈ +48 eV per σ); trained from scratch it keeps 0.72–0.78 of the backbone's
  PCHIP-relative skill on 4/4 cut splits (registered bar 0.70).
- Median: a hold-or-jump filter with Student-t noise at the TRAIN bulk scale reaches 0.906–0.920×
  persistence on 8/8 cells (§8bp). The backbone itself is a mean estimator; its aggregate skill is
  a tail statistic (§8ax).

**Claim 2 — rotation: a lower mean error too, carried by the CES history alone.**
- B.27c two-level bootstrap over eight new partitions, vs PCHIP: +0.313 [+0.072, +0.701] (cut),
  +0.319 [+0.046, +0.683] (inclusive); vs persistence cut [+0.277, +0.763]; vs causal GP cut
  [+0.045, +0.701]; vs offline GP cut [−0.119, +0.635] (includes 0). The fold-level overlap-corrected
  interval vs PCHIP, [−0.021, +0.405], still includes 0 (§8by). The four-split PR4 gate was 1/4
  (cut) and 2/4 (inclusive) (§8ab); vs linear per split 1/4 in both populations (2026-11-11 read).
- Why the gain is history-only: momentum equation r(BES, dV_rot/dt) = −0.006,
  r(ECEI, dV_rot/dt) = −0.003 (§8ar). 100 Hz MC values equal a raw 2 MHz sample at the grid time on
  24/24 coils; 15 registered features of the raw Mirnov stream change `V_rot` skill by −0.128
  [−0.544, −0.105] (cut) / −0.164 [−0.473, −0.133] (inclusive), 0/3 test discharges, six training
  discharges (§8bd). NBI power is a step input: the band carried by a 300 ms moving average holds
  0.34 % of the 10 ms `V_rot` increment variance, so NBI is a lever only for forecasts of 100 ms
  and longer (§8bd addendum).
- Median: the backbone's median row is worse than persistence (4.3 vs 3.7 km/s, cut, §8ax); the
  physics filter's ratio to persistence is 0.94–1.00 (0.938–1.003 per split) with the interval including 1 on 3/4 splits;
  the T_i / V_rot scatter ratio is the √2 of a Gaussian line fit on 8/8 cells — the 10 ms photon
  floor (§8bp, §8bq–§8bt). The remaining 10 ms lever is per-exposure CES fit uncertainty, which the
  641 shot files do not contain.

**Closed axes.** Model size (26× width sweep flat, §8aa), operator family (§8ag), 4-init ensemble
+0.02 (§8aw), context beyond 50 ms (§8al), Huber loss cut-conditional (§8bu), soft gate negative
(§8bg). **Limits.** One device and one campaign; the offline claim is bounded above by the offline
GP; the value cut is a proxy for fit quality.

**Paper.** Only the 8-page version is kept: `docs/paper/short/main_short.tex` (English) and
`main_short_ko.tex` (Korean), numbers traced to this file. The 33/38-page full paper was removed on
2026-11-11 (last present in commit 2318fd9).

