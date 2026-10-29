# THESIS_RESULTS — KSTAR CES nowcasting vs. conventional interpolation

> **✅ FINAL CONFIRMED RESULT (robust, multi-seed).** With the AutoML-improved final model
> (GRU history-encoder + observation-masked multi-head attention per-target heads, `window = 4`),
> the thesis claim holds: across **4 independent held-out test splits** (seeds 42/1/7/123; 1/7/123
> never used in architecture selection), **`CES_TI` skill_vs_pchip = +0.19…+0.28 with a
> shot-clustered 95% CI excluding 0 every time (PASS)**; **`CES_VT` is n.s. on all four** — the
> `T_i` ↔ `V_rot` asymmetry. The model also beats every causal baseline (persistence / local AR).
> Retained limitation: skill measured on observed CES points only (MNAR optimistic bound); `window = 4`.

## Provenance — every number in this document (regenerated 2026-07-14)

All numbers below come from **one coherent, reproducible family of checkpoints**. Two traps make
this non-obvious, so they are stated up front:

1. **`ces_prediction/model.py` can no longer load ANY saved checkpoint.** The AutoML loop rewrote it
   into a Transformer history encoder. The architecture behind every number here is the archived
   snapshot **`ces_prediction/model_iter009.py`** (final model) and
   **`ces_prediction/model_iter002.py`** (the earlier iter2 baseline, §6) — both byte-identical
   copies of the AutoML archive, moved into the repo on 2026-08-09 and pinned by SHA-256 in
   `tests/test_architecture.py`. Inject the
   right one as the `model` module (`sys.modules["model"] = …`) before importing the harness.
2. **The old `data/.improve_final_out/` and `data/.ms_out_*/` weights are stale** — they reproduce
   `CES_TI` but not `CES_VT` (e.g. seed 42 gives skill +0.056, not +0.161). They are NOT the
   checkpoints that produced the previously-recorded metrics and must not be used.

| What | Source |
|---|---|
| Final model, seed 42 | `data/.vt_repro_out/` |
| Final model, seeds 1 / 7 / 123 | `data/.vt_repro_ms_{1,7,123}/` |
| Ablations (`no_fast`, `no_history`) | `data/.vt_repro_ab_{no_fast,no_history}/` |
| iter2 baseline (§6 progression) | `data/.final_out/` (reproduces its recorded numbers exactly) |
| Splits | `data/.improve_split/w4` (seed 42), `data/.ms_split_{1,7,123}` |
| Bootstrap | `bootstrap_summary.json` in each dir (shot-clustered, B = 10,000, seed 12345) |

**Evaluation conventions.** Training keeps held ("stuck") CES values (`CES_DROP_STUCK_TARGETS=0`,
the `train.py` default). The **headline** evaluation also keeps them (=0, 34,644 test samples), and
the **genuine-only** evaluation (=1, held values dropped) is reported separately in §1 as a
robustness check. `peak` numbers (§5) are on the **validation** split; everything else is TEST.

---

## 1. Data-quality correction — held ("stuck") CES_VT values

A late audit found that **~54% of all observed `CES_VT` values are held/forward-filled**: a reading
bit-identical to the previous observed value within the same contiguous time block (runs up to 1214
rows; 499/641 shot files affected). These are not real measurements — `CES_VT`'s native cadence is
slower than the row cadence and the value is carried forward. `CES_TI` is essentially unaffected
(**0.0%** held).

- **OVERTURNED 2026-07-30 (see §8c): held values DO contaminate training.** The earlier
  "masking held at train time does not help" conclusion rested on a single-seed val-metrics
  A/B (`data/.ab_stuck_B`). A proper 4-seed paired experiment (identical architecture and
  file split, `CES_DROP_STUCK_TARGETS=1` as the only variable) shows held-free training beats
  held-kept training on `CES_VT` on **all 4 seeds (3/4 CIs exclude 0)** with no `CES_TI` cost.
  The historical convention (train keeps held, evaluation drops them) still describes every
  pre-§8c number in this document.
- **`CES_TI` is fully robust.** On genuine-only evaluation the final model still **PASSes on all four
  seeds**.
- **`CES_VT` physical RMSE is deflated ~35–55% by the held values** (held targets have ≈0 baseline
  error and drag the reported RMSE down).

| seed | reported (stuck-incl) VT RMSE | **genuine VT RMSE** | VT n (reported → genuine) | VT skill (genuine) | gate | TI skill (genuine) | gate |
|---|---:|---:|---:|---:|---|---:|---|
| 42  | 22.5 | **35.0** | 27,437 → 10,729 | +0.203 | n.s. | +0.179 | **PASS** |
| 1   | 24.7 | **34.8** | 29,793 → 13,660 | +0.162 | **PASS** | +0.197 | **PASS** |
| 7   | 30.0 | **43.5** | 32,016 → 14,698 | +0.100 | n.s. | +0.280 | **PASS** |
| 123 | 32.9 | **46.5** | 30,475 → 14,126 | +0.183 | n.s. | +0.263 | **PASS** |

The `T_i` ↔ `V_rot` asymmetry conclusion is unchanged: on genuine data `CES_VT` is n.s. on 3/4 seeds
(PASS on seed 1 only). **Any `CES_VT` physical-RMSE figure elsewhere in this document is the
stuck-deflated value; the genuine RMSE is ~35–47.**

### 1.1 Missingness ledger — full 641-file census

Regenerate with `python ces_prediction/analyze_data_evidence.py`. Counting NaN alone understates
`CES_VT` badly, so **"`CES_VT` is 24% missing" must not be quoted on its own**: the honest figure
for how much of the grid carries no independent `CES_VT` information is **65.0%**.

| | `CES_TI` | `CES_VT` |
|---|---:|---:|
| 10 ms grid rows (641 shots) | 247,207 | 247,207 |
| ① NaN-missing | 20,216 (8.2%) | 59,107 (23.9%) |
| ② held / padded (bit-identical to previous observation) | 1 (0.0%) | 101,604 (41.1%) |
| **effective missing ① + ②** | **20,217 (8.2%)** | **160,711 (65.0%)** |
| independent observations | 226,990 (91.8%) | 86,496 (35.0%) |
| held as a share of observed values | 0.0% | 54.0% |
| shot files containing held runs | 1 / 641 | 499 / 641 |
| held run length (median / max) | 2 / 2 rows | 10 / 1,214 rows |

This reproduces the §1 audit (54% of observed, 499/641 files, runs to 1,214) from the raw CSVs.

---

## 2. Question, claim, and the deliberately-hard bar

**Question.** Can a multimodal *nowcasting* model — predicting the low-time-resolution KSTAR CES
targets `[CES_TI (ion temperature), CES_VT (toroidal rotation)]` at a target 10 ms timestep from
**simultaneous fast diagnostics (BES / ECEI / MC) plus past CES history** — recover CES information
beyond what temporal interpolation of CES alone can recover?

**The bar.** The model is benchmarked against *offline* CES-only interpolation (linear, monotone-cubic
PCHIP, local AR) that is **allowed to use both PAST and FUTURE CES samples** around the target step.
The model sees only fast diagnostics at the target step and *past* CES history. Beating a baseline
that even sees the future is strong evidence that the fast diagnostics carry CES-relevant information
that temporal interpolation cannot recover. (A model that beats future-using interpolation a fortiori
beats any causal baseline.)

**Claim.** The model beats conventional past+future interpolation for `CES_TI`; `CES_VT` is expected
*not* to win, and that non-win is itself the scientific finding — the `T_i` ↔ `V_rot` asymmetry (fast
diagnostics carry core-temperature information at 10 ms but little direct toroidal-rotation
information).

---

## 3. Method

### 3.1 Held-out three-way split (selection isolated from the headline)
`train.py` splits at the **file (shot) level** into **train / val / test**. The TEST split is reserved
before AutoML and is **never read by the search loop** — model selection happens on validation only,
so the test numbers carry no winner's-curse bias.

- TEST evaluation population (seed 42): **34,644 scored samples across 96 shots**.
- Observed per target after the per-target keep mask: **`CES_TI` n = 32,716**, **`CES_VT` n = 27,437**.
- Pre-registered TEST floor (PR3: ≥ 15 test shots and ≥ 3,000 observed `CES_TI` samples) is **met**.

### 3.2 Metric
All errors are denormalized to **physical CES units** and computed **per target**. The primary score
is the Murphy (1988) skill against the pre-registered headline baseline PCHIP:

```
skill_vs_pchip = 1 − MSE_model / MSE_pchip
```

Every arm (model + all baselines) is scored on the **identical** `(file, row_index)` sample set and
the **same per-target keep mask**. Baselines read CES neighbors from the in-memory filtered file
arrays; the target's own value at `row_index` is **excluded** (no leakage), and interpolation
**refuses across ≥ 0.5 s gaps**, falling back to persistence (PR2).

### 3.3 Shot-clustered paired bootstrap
Adjacent CES rows within a discharge are strongly correlated, so the **shot** is the resampling unit:
per-sample paired errors `SE_model − SE_pchip` are aggregated by shot and **whole shots are resampled
with replacement** (10,000 resamples, seed 12345). A 95% skill CI **excluding 0 in the model's favor
= PASS** (PR4). This CI reflects genuine **shot-to-shot generalization**, not within-shot
pseudo-replication.

### 3.4 Pre-registration (fixed before viewing TEST numbers)
- **PR1** — headline baseline is **PCHIP** (chosen a priori for ELM/sawtooth robustness); the full
  ladder {persistence, linear, PCHIP, AR} is also reported.
- **PR2** — interpolation predicts at every observed target the model is scored on; where no future
  observed neighbor exists in-window it falls back to persistence, so no arm is thinned.
- **PR3** — TEST reserved before AutoML; floor ≥ 15 shots and ≥ 3,000 observed `CES_TI` samples.
- **PR4** — shot-clustered paired bootstrap; 95% CI excluding 0 = PASS. Split-seed variation is a
  secondary stability check, never pooled into the headline CI.

---

## 4. Results (held-out TEST)

### 4.1 Headline — skill vs. PCHIP across 4 independent test splits

| Target | seed | skill_vs_pchip | 95% CI (shot-clustered) | PR4 gate |
|---|---|---:|---|---|
| **CES_TI** | 42  | **+0.257** | [+0.118, +0.360] | **PASS** |
| **CES_TI** | 1   | **+0.194** | [+0.080, +0.279] | **PASS** |
| **CES_TI** | 7   | **+0.263** | [+0.167, +0.334] | **PASS** |
| **CES_TI** | 123 | **+0.280** | [+0.140, +0.359] | **PASS** |
| CES_VT | 42  | +0.154 | [−0.413, +0.336] | n.s. |
| CES_VT | 1   | +0.109 | [−0.056, +0.218] | n.s. |
| CES_VT | 7   | +0.065 | [−0.693, +0.278] | n.s. |
| CES_VT | 123 | +0.127 | [−0.462, +0.244] | n.s. |

`CES_TI` **passes PR4 on all four independent test splits** — seeds 1/7/123 were never used in
architecture selection, so this is a genuine out-of-selection replication. `CES_VT` is n.s. on all
four: the point estimate is consistently positive but the shot-clustered CI always includes 0.

### 4.2 Per-target RMSE ladder (seed 42, physical units)

Lower is better. `ar_local` is past-only (causal); persistence is the last observed value; linear and
PCHIP use past+future neighbors. Baseline RMSEs are deterministic (model-independent).

**CES_TI** (n = 32,716; future-neighbor fraction = 0.996)

| Arm | Access | RMSE | vs. PCHIP |
|---|---|---:|---|
| **Model (nowcaster)** | fast diagnostics + past CES | **372.32** | **model better** |
| Linear interpolation | past + future CES | 422.66 | model better |
| PCHIP interpolation *(headline)* | past + future CES | 431.81 | model better |
| Persistence | last observed CES | 487.31 | model better |
| AR (local) | past CES only | 1005.66 | model better |

**CES_VT** (n = 27,437; future-neighbor fraction = 0.909; RMSE stuck-deflated — see §1)

| Arm | Access | RMSE | vs. PCHIP |
|---|---|---:|---|
| **Model (nowcaster)** | fast diagnostics + past CES | **22.53** | model better (n.s.) |
| Linear interpolation | past + future CES | 24.01 | model better |
| PCHIP interpolation *(headline)* | past + future CES | 24.49 | model better |
| Persistence | last observed CES | 27.77 | model better |
| AR (local) | past CES only | 57.23 | model better |

The model has the **lowest RMSE in the full ladder for both targets**, and beats the causal baselines
(persistence, AR) by a large margin.

### 4.3 Gap-stratified summary (seed 42)

The data are overwhelmingly small-gap: Δt ≤ 15 ms holds 31,966 / 32,716 (`CES_TI`) and
26,938 / 27,437 (`CES_VT`) of observed samples, so the first row drives the aggregate.

**CES_TI**

| Δt bin | n | RMSE model | RMSE PCHIP | skill vs PCHIP |
|---|---:|---:|---:|---:|
| (0, 15] ms | 31,966 | 324.47 | 372.57 | **+0.242** |
| (15, 25] ms | 520 | 1188.41 | 1399.06 | **+0.278** |
| (25, 35] ms | 140 | 1490.81 | 2050.63 | **+0.471** |
| (35, 55] ms | 33 | 527.73 | 522.99 | −0.018 |
| (55, 105] ms | 12 | 1730.49 | 1815.72 | +0.092 |
| (105, ∞) ms | 45 | 1324.19 | 419.13 | −8.98 |

**CES_VT**

| Δt bin | n | RMSE model | RMSE PCHIP | skill vs PCHIP |
|---|---:|---:|---:|---:|
| (0, 15] ms | 26,938 | 15.37 | 18.51 | **+0.310** |
| (15, 25] ms | 357 | 118.96 | 121.59 | +0.043 |
| (25, 35] ms | 97 | 148.36 | 139.23 | −0.135 |
| (35, 55] ms | 14 | 31.44 | 69.02 | +0.792 |
| (55, 105] ms | 1 | 6.06 | 1.34 | −19.3 (n = 1, ignore) |
| (105, ∞) ms | 30 | 109.37 | 2.16 | −2555 |

The advantage concentrates in the **small-gap regime**, which is where the overwhelming majority of
real targets live and where nowcasting matters. At Δt > 105 ms PCHIP (with a genuine future anchor and
a near-trivial interpolation problem) crushes the model — but these are tiny bins (45 TI / 30 VT
samples) with no per-bin CIs and should not be over-interpreted.

---

## 5. Where the model earns its skill

### 5.1 High-variability ("peak") regions — validation split

`peak` rows are selected by an **input-only** proxy (large neighbor-bracket slope and/or high local
CES-neighbor variance, **excluding** the target's own value at `row_index`), so the selection is not
circular.

| Target | global skill_vs_pchip | **peak** skill_vs_pchip | 95% CI (peak) | gate |
|---|---:|---:|---|---|
| CES_TI | +0.272 | **+0.702** | [+0.503, +0.851] | **PASS** |
| CES_VT | +0.131 | **+0.438** | [+0.068, +0.726] | **PASS** |

Interpolation is near-optimal in the smooth bulk; the model's value is concentrated in the **active**
regions. Notably `CES_VT`, which is n.s. globally, **passes in the peak subset**, so the
`T_i` ↔ `V_rot` asymmetry is *regional*, not absolute.

A per-shot case study of exactly this (KSTAR **#31815**, held-out TEST) is in the presentation: the
model tracks the transients while PCHIP overshoots at every spike (TI skill +0.36, peak +0.63;
VT +0.16). Across the whole test split the model beats PCHIP on `CES_TI` in **43 of 89** shots — the
win is real but concentrated, not uniform.

### 5.2 Input-modality ablation (validation, `skill_vs_persistence`)

| Ablation | CES_TI | CES_VT |
|---|---:|---:|
| Full (history + fast + time) | +0.428 | +0.296 |
| `no_fast` (history only) | +0.458 | +0.295 |
| `no_history` (fast only) | **+0.372** | **−0.642** |

This is the mechanism behind the asymmetry. With **fast diagnostics alone** (no CES history), the
model still reaches **+0.372** on `CES_TI` — the fast diagnostics genuinely carry ion-temperature
information at 10 ms (collisional e–i coupling: ECEI sees `T_e`, BES sees `n_e`). On `CES_VT` the same
fast-only model scores **−0.642**, i.e. *worse than persistence*: toroidal rotation is driven by
unobserved NBI torque, and the Mirnov coils are aliased at the 10 ms grid, so essentially all `V_rot`
information comes from the past CES history.

---

## 6. Honest progression: n.s. → significant

The earlier pre-improvement baseline (**iter2**: GRU history-encoder with a plain final-hidden-state
readout and per-target heads; `data/.final_out/`) was **not significant** on the same TEST split. The
AutoML loop's selected change — an **observation-masked multi-head attention readout over the GRU
sequence** — is what moved it across the line:

| Model | CES_TI skill_vs_pchip | 95% CI | PR4 gate |
|---|---:|---|---|
| iter2 baseline (`data/.final_out/`) | +0.088 | [−0.221, +0.323] | n.s. |
| **Final model** (`data/.vt_repro_out/`) | **+0.257** | **[+0.118, +0.360]** | **PASS** |

The decisive methodological point: the loop selected on **clean validation skill vs. interpolation**,
not on the augmented training/val loss. Selecting on val loss would not have found this.

---

## 7. Honest conclusion

1. **The model decisively beats every causal baseline.** Against persistence and a past-only AR
   reference the model is far ahead on both targets (`CES_TI` 372 vs 487 / 1006). For any online /
   real-time setting — where future CES is by definition unavailable — the model is the clear winner.
2. **`CES_TI` significantly beats offline future-using interpolation.** +0.19…+0.28 with the
   shot-clustered 95% CI excluding 0 on **all four** independent test splits, and still PASSing on
   genuine-measurement-only evaluation. PR4 passes.
3. **`CES_VT` ties interpolation globally (n.s.).** The point estimate is positive on all four seeds
   but the CI always includes 0. This is not hidden or spun as a win. Note that `CES_VT` is also the
   *unstable* estimate: a slightly different checkpoint of the same architecture can swing its point
   estimate substantially, which is itself consistent with a low-power, wide-CI quantity.
4. **The `T_i` ↔ `V_rot` asymmetry is real, predicted, and mechanistically explained.** Fast
   diagnostics carry core-temperature information at 10 ms (fast-only `CES_TI` = +0.372) but carry
   essentially no direct toroidal-rotation information (fast-only `CES_VT` = −0.642, worse than
   persistence). NBI torque — the actual driver of `V_rot` — is **not in the dataset at all**, and
   Mirnov is aliased at the 10 ms grid. The non-win on `V_rot` is reported as the scientific finding,
   not as a failure.

---

## 8. Limitations

- **Statistical power (~96 test shots).** The shot is the correct independent unit and there are only
  ~96 (TI) / ~91 (VT) of them. This is the binding constraint on every significance call, and it is
  why `CES_VT` cannot be resolved.
- **Heavy-tailed errors.** Per-shot squared-error differences are heavy-tailed; a few discharges
  dominate the bootstrap spread.
- **MNAR optimistic bound.** Skill is measured **only on observed CES points**. The points where CES
  happens to be observed may be easier than the unobserved ones.
- **Neighbor-access asymmetry.** Interpolation baselines use full-shot CES neighbors while the model
  uses only a `window_size = 4` history. This is intentional (it is the point of the claim) but limits
  direct interpretation.
- **Single architecture, single window.** Architecture/window sensitivity is not characterized.
- **Thin large-gap bins.** Δt > 25 ms bins have tens of samples or fewer and no per-bin CIs; their
  skill values (e.g. −2555) are unstable and not load-bearing.
- **CES fit-failure artifacts.** `CES_TI` values above ~3 keV (global p99 = 2,089 eV; max 14,984 eV)
  are spectral-fit failures, not physics. They survive in the evaluation population and inflate the
  large-gap bins.

---

## 8b. Data-level evidence for two expected rebuttals

Regenerate all of §8b with `python ces_prediction/analyze_data_evidence.py` (full 641-file scan).

### 8b.1 "Why PCHIP as the headline baseline?"

PCHIP is **weaker than linear** on this data (`CES_TI` RMSE 431.81 vs. 422.66), so the honest
defence is not "PCHIP is the strongest bar" — it is that the choice was pre-registered (PR1, fixed
before any TEST number was seen), the full ladder is reported, and **swapping the headline baseline
for the stronger `linear` leaves the conclusion intact**. From `bootstrap_summary__test_stuck0.json`:

| seed | `CES_TI` vs PCHIP | `CES_TI` vs **linear** |
|---|---|---|
| 42 | +0.257 [+0.118, +0.360] PASS | +0.224 [+0.088, +0.326] **PASS** |
| 1 | +0.194 [+0.080, +0.279] PASS | +0.163 [+0.052, +0.247] **PASS** |
| 7 | +0.263 [+0.167, +0.334] PASS | +0.242 [+0.151, +0.311] **PASS** |
| 123 | +0.280 [+0.140, +0.359] PASS | +0.250 [+0.116, +0.324] **PASS** |

Headline evaluation: **4/4 PASS against linear**. Genuine-only evaluation: 3/4 PASS (seed 42 is
n.s. at [−0.021, +0.267]). Why PCHIP lost to linear here: Δt ≤ 15 ms covers 97.7% of scored
samples, so the cubic has no room to help and only adds derivative-estimation noise.

### 8b.2 "Is the Mirnov signal really aliased?"

Lag-1 autocorrelation within contiguous time blocks, same 100 Hz grid for every diagnostic:

| diagnostic | blocks | mean r | median r | share with \|r\| < 0.1 |
|---|---:|---:|---:|---:|
| BES | 5,886 | +0.568 | +0.647 | 7.6% |
| ECEI | 2,598 | +0.572 | +0.653 | 9.2% |
| **MC (Mirnov)** | 1,308 | **−0.009** | **−0.004** | **82.0%** |

BES/ECEI are temporally continuous; MC is white noise on the same grid. The precise statement is
not "aliasing" in the loose sense but: **`dB/dt` oscillates at the kHz mode frequency and was
decimated to 100 Hz without an anti-aliasing filter, so successive samples have random relative
phase and the amplitude/mode-number information is destroyed before the model sees it.** This is
consistent with the earlier negative result that derived MC features (integral, PCHIP integral,
|MC|, rolling RMS) never helped — information already lost cannot be recovered downstream.
Recovering it requires per-window RMS / band power / mode number computed from the **raw** kHz
Mirnov timeseries.

### 8b.3 "Doesn't T_e proxy the NBI heating, and hence the torque?"

A physically reasonable chain (NBI → electron heating → `T_e`), and the **first half is confirmed**
by the data — which is exactly why the null on the second half is informative. ECEI channel mean is
the `T_e` surrogate; 538 shots with ≥ 20 observations of each target:

| relation (between shots) | Pearson r | p |
|---|---:|---:|
| `T_e` ~ `CES_TI` | **+0.353** | 2.9e−17 |
| `T_e` ~ `CES_VT` | +0.024 | 0.58 |
| `T_e` ~ \|`CES_VT`\| | +0.001 | 0.98 |
| `T_e` variability ~ \|`CES_VT`\| | −0.026 | 0.55 |
| BES variability ~ \|`CES_VT`\| | −0.059 | 0.17 |

Within shots the same split holds: `T_e`~`CES_TI` averages r = +0.246 with a **consistent sign**
(42.7% of blocks exceed |r| = 0.3), while `T_e`~`CES_VT` averages +0.006 — the sign itself is
random (14.8%). So `T_e` does track the heating level, but that information does not reach `V_rot`.
The break is **power ≠ torque**: torque depends on beam energy, tangency radius and injection
geometry (so it separates from power), and rotation is governed by momentum transport and edge
braking (NTV, error fields, wall drag) rather than by local `T_e`. This strengthens rather than
weakens the §7 conclusion that the missing NBI torque channel is the real lever for `CES_VT`.

---

## 8c. Stuck-free training experiment (2026-07-30) — held values DO hurt training

**Motivation (raised by 승상님).** The final model's physics story — "`V_rot` is strongly
persistent on the 10 ms grid, so past CES history is the dominant `V_rot` source" — was learned
from data in which 54% of observed `V_rot` values are forward-filled holds. Held values sat in
the training targets AND in the `ces_history` input channels, teaching the model that copying
history is near-optimal: the persistence assumption was partly an artifact of the instrument
padding, i.e. circular. The §6.8 A/B that "settled" this was a single-seed val-metrics check —
below this project's own ≥4-seed-paired bar for any `CES_VT` claim.

**Design** (`ces_prediction/experiments/stuckfree/run_stuckfree.py`; one controlled variable):
- Architecture: the pinned thesis model `model_iter009.py`, identical in both arms.
- Treatment: train **and** evaluate with `CES_DROP_STUCK_TARGETS=1` — held values are NaN'd at
  dataset load, removing them from (a) supervision targets, (b) `ces_history` values and
  observation flags, (c) normalization stats, (d) interpolation-baseline anchors.
- Control: the trusted `data/.vt_repro_*` family (held kept at train time, genuine-only eval).
- File-level split **pinned to the control manifest** (`CES_FILE_SPLIT_FROM`, added to
  `train.py`): drop_stuck can shrink the valid-file list, and the seeded shuffle would
  repartition and leak control test shots into training (observed on seeds 7/123 before the
  fix). Manifest-unlisted files (capped out in the control run) are excluded from training.
- Paired scoring: same population row-for-row (shot + `dt_ms` bit-identity; se_pchip within
  float32 stats-round-trip tolerance), shot-clustered paired bootstrap B = 10,000.

**Result — held-free training wins `CES_VT` on every seed (verdict: KEEP_CANDIDATE):**

| seed | paired `CES_VT` (held-free vs held-kept) | paired `CES_TI` | own `CES_TI` vs PCHIP | own `CES_VT` vs PCHIP |
|---|---|---|---|---|
| 42  | **+0.030 [+0.001, +0.052] PASS** | +0.072 [+0.012, +0.137] PASS | +0.238 PASS | +0.226 n.s. |
| 1   | **+0.047 [+0.012, +0.081] PASS** | −0.016 n.s. | +0.184 PASS | **+0.202 PASS** |
| 7   | +0.048 [−0.015, +0.152] n.s. | +0.016 n.s. | +0.291 PASS | +0.143 n.s. |
| 123 | **+0.032 [+0.006, +0.057] PASS** | −0.057 n.s. | +0.221 PASS | +0.210 n.s. |

- `CES_VT`: **4/4 point estimates positive, 3/4 shot-clustered CIs exclude 0, 0 against;
  mean paired skill +0.039** (≈4% MSE reduction from the data fix alone).
- `CES_TI`: no consistent effect (mean ≈ +0.004; one seed significantly better, none worse) —
  removing held `V_rot` from the history channel does not hurt the `T_i` pathway.
- Own-vs-PCHIP: `CES_TI` still PASSes 4/4 (+0.184…+0.291). `CES_VT` point estimates improve on
  every seed vs the control's genuine numbers (42: +0.203→+0.226, 1: +0.162→+0.202,
  7: +0.100→+0.143, 123: +0.183→+0.210) with the same PR4 pattern (seed 1 PASS only) — the
  headline "`V_rot` vs interpolation is n.s." conclusion stands, but the model itself is
  strictly better.

**Interpretation.** The instrument's forward-fill padding was actively teaching the model to
lean on history copying: remove it and the same architecture extracts more genuine `V_rot`
signal. This partially de-circularizes the persistence story — `V_rot`'s reliance on history is
real physics AND was inflated by the artifact; the honest statement quantifies both. Training
convention for future runs: **`CES_DROP_STUCK_TARGETS=1` at train time is the better default**
for `CES_VT`-relevant work.

---

## 8d. Sequence reframing experiment (2026-07-30) — full-grid LSTM + masked loss

**Motivation (승상님).** The window pipeline's irregular time axis is self-inflicted: rows
without an observed target were dropped. The fast diagnostics are 100% dense on the 10 ms
grid, so a causal LSTM can consume EVERY row (unlabelled rows still provide input context)
with sparse labels handled purely in the loss (per-target observed mask) — eliminating the
window, the combinatorial temporal-subset augmentation, and the sample caps entirely.

**Implementation** (`ces_prediction/experiments/seq/`): 0.34M-param unidirectional LSTM;
per-step features = z-scored fast channels + Δt + per-target carry-forward/staleness/has-obs;
blocks split at ≥ 0.5 s gaps; masked MSE + the train.py TI penalty; early stopping on val.
Evaluation forks compare_baselines.py with a (file, time)-keyed prediction lookup — same
population, baselines, and npz schema; the harness round-trip reuses the control stats so
se_pchip stays bit-identical. Windows note: CUDA teardown fastfails (0xC0000409) after all
artifacts are saved; the runner trusts fresh artifacts over the exit code.

**Results — three paired comparisons close the loop (4 seeds each, shot-clustered):**

| comparison | CES_TI | CES_VT |
|---|---|---|
| seq (held-kept) vs iter009 (held-kept) | +0.078 / +0.030 / +0.011 / **+0.049*** — 4/4 positive, mean +0.042, own PR4 4/4 | −0.103* / −0.048* / −0.061 / +0.017 — 2/4 significantly worse |
| seq (held-free) vs iter009 (held-kept) | +0.071 / +0.019 / +0.017 / **+0.049*** — mean +0.039 | −0.013 / −0.021 / −0.016 / +0.010 — **no significant difference** |
| seq (held-free) vs iter009 (held-free) | −0.001 / +0.035 / +0.002 / **+0.100*** | **−0.044* / −0.072* / −0.067* / −0.023* — 4/4 significantly worse** |

(* = CI95 excludes 0; seed order 42/1/7/123.)

**Interpretation.**
- The framing wins `CES_TI` in every comparison (mean ≈ +0.04, own PR4 4/4 PASS) with a
  simpler pipeline — the reframing is validated for `T_i`.
- The `CES_VT` deficit decomposes exactly into the two suspected causes: (1) **held
  contamination** — held-free training removes every significant degradation (row 1 → row 2);
  (2) **missing routing** — under identical held-free data (row 3) the shared-encoder LSTM
  still loses `V_rot` on 4/4 seeds, confirming iter009's V_rot head design (fast diagnostics
  blocked, observation-masked history attention) carries its `V_rot` advantage.
- **Next controlled change: seq v2 = this framing + the iter009 V_rot routing (V_rot head fed
  only carry-forward/staleness/time state), trained held-free.** Combines the two confirmed
  wins (`T_i` +0.04 from the framing, `V_rot` +0.04 from held-free) while restoring the one
  protection the minimal seq model lacked.

Artifacts: `data/.seq_lstm_s*`, `data/.seq_sf_lstm_s*` (`paired_vs_iter009.json`,
`paired_vs_sf_iter009.json`), summaries `data/.seq_summary.json`, `data/.seq_sf_summary.json`.

---

## 8e. Continuous-time history encoders (2026-07-30) — verified negative

> **Code retired 2026-08-09.** The batch was a verified negative and the paper no longer
> claims it, so `experiments/ct/` and its plan doc were removed from the tree (recover from
> git history if the question is ever reopened). Its two genuinely shared pieces were kept
> and promoted: `experiments/runner_common.py` (split/control/env constants, `run_step`) and
> `experiments/paired_model_compare.py`. **This record stands as the result** — the verdict
> below is why no one should re-run it at `W = 4`.

Pre-registered batch (`experiments/ct/run_ct_experiments.py`, since removed; plan in
docs/연속시간_모델_실험계획.md, since removed): swap ONLY the history encoder for four
continuous-time variants — exponential-decay GRU (GRU-D형), ODE-RNN, Neural CDE, Δt-exact
diagonal SSM — same 4 splits, paired shot-clustered bootstrap vs iter009, `CES_TI` verdict.

| variant | verdict | paired `CES_TI` (42/1/7/123) | favor / against |
|---|---|---|---|
| ct_decay | **SIGNAL** | +0.022* / −0.023 / −0.005 / +0.023* | 2/4 · 0/4 |
| ct_odernn | FAIL | +0.039 / +0.027 / −0.008 / +0.009 (all n.s.) | 0/4 · 0/4 |
| ct_ncde | FAIL | −0.032 / −0.055* / +0.015 / +0.024 | 0/4 · 1/4 |
| ct_ssm | FAIL | +0.045 / −0.052* / +0.003 / −0.034 | 0/4 · 1/4 |

No KEEP_CANDIDATE. The only signal is the simplest variant (exponential-decay GRU), and it
flips sign on 2/4 seeds; the more expressive encoders (NCDE, SSM) each produce a
significantly-worse seed. On a regular 10 ms grid with `window = 4`, evolving the hidden
state through real Δt adds no measurable information — consistent with the day's larger
lesson (§8c–§8d): **the data treatment (held removal) and the framing (full-grid masked
loss) moved the needle; architecture micro-variants did not.** Do not revisit CT encoders
at this window size; if ever revisited, do it on long sequences (seq framing) where Δt
structure actually varies.

---

## 8f. Window sweep (2026-08-04) — one past observation is enough; `W = 4` is not justified

**Motivation (외부 피드백).** `CES_WINDOW_SIZE = 4` was an empirically-placed default with no
stated justification — "2나 3에서도 동작하는데 왜 4인가"에 답할 근거가 없었다. This sweep replaces
the assertion with a curve and an explicit selection rule.

**Design** (`ces_prediction/experiments/window_sweep/run_window_sweep.py`; plan in
docs/설명성_피드백_실험계획.md §A). 24 independent runs = **W ∈ {2, 3, 4, 6, 8} × seeds {42, 1, 7, 123}**
plus a **history-0** point × 4 seeds. Model = the published `model_iter009.py`, swapped in per run.
Metric = each run's own held-out TEST `skill_vs_pchip` (`compare_baselines.py`) + PR4 shot-clustered
bootstrap PASS. Every point uses the same 10-epoch / batch-512 / lr-1e-3 protocol; only `W` varies.

Three controls make the points comparable, each verified on the real data rather than assumed:

- **File-level split is window-invariant** — probed directly at W = 2/4/8 × 4 seeds: identical
  valid-file list and identical train/val/test partitions. Confirmed post-hoc in the summary:
  **all 24 runs evaluate the same 96 test shots per seed** — and the same 96 as the held-kept
  pass, so §8f-R's contrast is on one population.
- **Per-shot sample cap `CES_MAX_SAMPLES_PER_FILE = 500`** (new in `train.py`, default off).
  Temporal-subset augmentation explodes combinatorially with `W` — the dataset holds 240k samples
  at W = 2 but **30.1M at W = 8**, with per-file counts up to 187k — so without a per-shot cap the
  seeded 200k train subset would be dominated by a few long-block shots *more strongly the larger
  W is*, confounding the very comparison being made. `CES_MAX_TEST_SAMPLES = 48000` (> 96 shots ×
  500) keeps the global test cap non-binding, so no shot is ever dropped from a test manifest.
- **Evaluation population is near-constant.** A clean full-window sample needs `W` contiguous rows,
  so `n` shrinks with `W` — but only marginally: seed 42 goes 33,900 → 33,288 eval samples
  (−1.8%) from W = 2 to W = 8. The history-0 point has *byte-identical* population to W = 4.

**history-0 point.** `W = 1` is impossible (`dataset.py` requires `window_size >= 2`), so history-0
is `CES_ABLATE=no_history` at W = 4: the model is **trained and evaluated** with the `ces_history`
tensor zeroed while the sensor window and time features stay at W = 4. This isolates history more
cleanly than W = 1 would (which would also shrink the sensor window). `compare_baselines.py` was
extended to honour `CES_ABLATE` **for the model's inputs only** — the persistence/PCHIP baselines
still read the real history, so this point's skill is measured against the identical bar.

**Data treatment — held-free, per §8c.** Held/forward-filled CES values are removed at **train
time as well as eval** (`CES_DROP_STUCK_TARGETS=1`). This is not cosmetic: held values are literal
copies of the previous reading, so under contamination a longer window merely supplies more copies
of the same number — the exact signature of "a longer trajectory helps rotation". The first pass of
this sweep left training held-kept and produced a `CES_VT` conclusion that this correction
overturns; that pass is retained as a reference contrast in §8f-R below.

**Result — the curve** (`docs/presentation/figures/fig_window_sweep.png`,
`data/.wsweep_hf_summary.json`):

| history | W | `CES_TI` mean | per-seed (1/7/42/123) | PASS | `CES_VT` mean | per-seed (1/7/42/123) | PASS |
|---|---|---|---|---|---|---|---|
| 0† | 4 | **−0.026** | −0.014 / −0.138 / −0.094 / +0.143 | 0/4 | **−0.783** | −1.235 / −0.544 / −1.039 / −0.312 | 0/4 |
| 1 | 2 | +0.238 | +0.219 / +0.258 / +0.181 / +0.296 | **4/4** | **+0.206** | +0.165 / +0.116 / +0.264 / +0.278 | 0/4 |
| 2 | 3 | **+0.246** | +0.184 / +0.289 / +0.226 / +0.286 | **4/4** | +0.203 | +0.224 / +0.122 / +0.235 / +0.230 | 1/4 |
| 3 | 4 *(incumbent default)* | +0.221 | +0.193 / +0.241 / +0.159 / +0.292 | 3/4 | +0.190 | +0.214 / +0.133 / +0.220 / +0.192 | 1/4 |
| 5 | 6 | +0.190 | +0.148 / +0.215 / +0.230 / +0.167 | 3/4 | +0.205 | +0.211 / +0.143 / +0.226 / +0.242 | 1/4 |
| 7 | 8 | +0.216 | +0.139 / +0.250 / +0.236 / +0.240 | **4/4** | +0.204 | +0.209 / +0.170 / +0.185 / +0.251 | **2/4** |

† history-0 = `no_history` ablation at W = 4 (see above).

**Reading 1 — history is essential, and its first observation carries essentially all of it.**
Removing previous-CES entirely puts `CES_TI` **below PCHIP** (−0.026, 0/4) and `CES_VT` at
**−0.78, i.e. ~1.8× PCHIP's MSE**. A single previous observation lifts both targets to their
maximum at once (`CES_TI` +0.238 4/4, `CES_VT` +0.206). So the fast diagnostics alone do not reach
interpolation parity; the model's entire margin comes from combining them with **one** past CES
sample.

**Reading 2 — the curve is flat from history = 1 onward, for BOTH targets.** `CES_TI` means span
0.190–0.246 (range 0.056) and `CES_VT` 0.190–0.206 (range **0.016**), while the *seed* spread
inside a single point is 0.08–0.13 (`CES_TI`) and 0.07–0.16 (`CES_VT`) — larger than the spread
across the whole curve. Neither target shows a trend, and the best point is at history = 2 for
`CES_TI` and history = 1 for `CES_VT`.

**Verdict — select `W = 2`.** This sweep exists to *choose* the window, not to defend the incumbent:
`W = 4` was never a decision, only `train.py`'s default with no recorded rationale, which is exactly
the criticism that prompted the experiment. The selection rule "smallest W that reaches plateau"
returns **W = 2** — both targets are already at plateau there, and `W = 4` is *below* the W = 2/3
values on both. Nothing in the data supports paying for a longer window on accuracy grounds.
`W = 3` is the runner-up (best `CES_TI` mean, +0.246, also 4/4) and is the choice if one wants a
little slack in the history slot; the coverage argument below is the only reason to go beyond that.

**The one real argument for a wider window is coverage, not skill.** `compare_baselines.py` scores
a sample only when window-persistence exists — an observed value **inside** the window. A W = 2
window has a single history slot, so a target whose adjacent row lacks that reading is dropped from
scoring entirely; a wider window can still reach back and produce a prediction. Total scored counts
barely move with `W` (`CES_VT` 52.1k → 53.2k), but the hard subset grows sharply:

| `CES_VT` (4 seeds pooled) | W=2 | W=3 | W=4 | W=6 | W=8 |
|---|---:|---:|---:|---:|---:|
| dt > 15 ms | 456 | 1,423 | 1,756 | 1,940 | 1,958 |
| dt > 45 ms | 14 | 23 | 27 | 99 | 135 |

So a wider window does not predict the long-gap samples *better* — it predicts **4–10× more of
them at all**. That is a coverage property, on a different axis from the skill curve, and it is the
only defensible reason to prefer W > 2.

**Honest caveats.** (1) These are *unpaired* independent runs; every difference along the curve sits
inside seed noise, so the claim is the **absence** of a window effect, not a ranking among windows.
(2) Because of the per-shot cap (`CES_MAX_SAMPLES_PER_FILE = 500`) the absolute skills are not
directly comparable to the §4.1 headline family, which had no such cap; the sweep is designed for
**within-curve** comparison, where every point shares it.

### 8f-R. Reference contrast — the same sweep with held-kept training

The first pass (`data/.wsweep_*`) is identical except that training kept held values. Evaluation was
genuine-only in both, and **the two sweeps score the same 96 test shots per seed**, so the arms are
directly comparable. Mean skill, held-free minus held-kept:

| history | W | `CES_TI` hf / kept (Δ) | `CES_VT` hf / kept (Δ) |
|---|---|---|---|
| 1 | 2 | 0.238 / 0.239 (−0.001) | 0.206 / 0.118 (**+0.088**) |
| 2 | 3 | 0.246 / 0.213 (+0.033) | 0.203 / 0.155 (**+0.048**) |
| 3 | 4 | 0.221 / 0.198 (+0.024) | 0.190 / 0.155 (**+0.035**) |
| 5 | 6 | 0.190 / 0.231 (−0.041) | 0.205 / 0.202 (+0.003) |
| 7 | 8 | 0.216 / 0.212 (+0.004) | 0.204 / 0.197 (+0.006) |

The `CES_VT` gain from removing held **decays monotonically with window size** (+0.088 → +0.048 →
+0.035 → +0.003 → +0.006), which is the mechanism in plain sight: with held present, a short
window's only history slot is often a forward-filled copy carrying no information, while a long
window still reaches a genuine observation. Held was therefore **penalising short windows**, not
rewarding long ones. Removing it lets W = 2 catch up completely, and the apparent "`V_rot` needs a
long history" slope of the first pass (+0.118 → +0.202, PASS 0/4 → 2/4) disappears. `CES_TI`, which
is ~0.0 % held, shows no systematic shift (Δ ranges −0.041…+0.033 with no consistent sign).

This is §8c's finding — held is a contaminant, not a harmless prior — reproduced along an axis it
was never tested on, and it is the reason the held-kept numbers are kept here only as a contrast.

---

## 8g. Large-gap regime vs. a CAUSAL baseline (2026-08-05) — not a weakness, it is interpolation's territory

**Motivation.** §4.3 and the window sweep both show `CES_TI` skill going negative in the wide-Δt
bins (−8.98 at Δt > 105 ms, seed 42), which reads as "the model collapses exactly where gap-filling
is hardest". The trap is the comparator: **PCHIP gets a future anchor**, so as the gap widens its
problem becomes *easier* (interpolate between two real observations) while ours becomes harder
(extrapolate from the past alone) — and **a real-time nowcaster cannot run PCHIP at all**. The
question that decides whether this is a model defect is: at large gaps, does the model still beat
the best baseline a deployed system *could* run — persistence?

**What was blocking it.** `compare_baselines.py` computed persistence per sample (`persist_phys`)
but never wrote it to `comparison_errors_test.npz`, so the causal comparison was unanswerable from
the saved artifacts.

**What was done** (no retraining):
1. One additive key in `compare_baselines.py`: `boot[f"{name}_se_persistence"]`. Pre-existing keys
   untouched (AC1).
2. `ces_prediction/experiments/largegap/rerun_compare_persistence.py` re-scores the trusted headline
   family (`data/.vt_repro_*`, seeds 42/1/7/123) under **both** treatments (`genuine` =
   `CES_DROP_STUCK_TARGETS=1`, `stuck0` = 0) from the saved weights + split manifests. It **verifies
   every pre-existing npz key reproduces bit-for-bit** before accepting the new one and never
   overwrites the originals (results land in `comparison_errors_test__{treatment}_persist.npz`).
   **All 8 runs: 12 keys bit-identical, +2 new.** Originals restored after each run.
3. `ces_prediction/experiments/largegap/analyze_largegap.py` **pools the 4 splits** (per-seed wide
   bins hold only tens of samples) and clusters the shot-clustered bootstrap on the **physical
   shot**, so a discharge appearing in two splits is one cluster, not two.

**Result (genuine evaluation, 4 splits pooled, B = 10,000).** Bold = CI excludes 0.

| Δt | n | shots | `CES_TI` vs PCHIP *(future-using)* | `CES_TI` vs **persistence** *(causal)* |
|---|---:|---:|---|---|
| ≤ 15 ms | 134,629 | 298 | **+0.262** [+0.20, +0.31] | **+0.407** [+0.36, +0.45] |
| (15, 25] | 2,987 | 259 | **+0.183** [+0.05, +0.30] | **+0.384** [+0.28, +0.48] |
| (25, 35] | 784 | 122 | **+0.402** [+0.21, +0.57] | **+0.564** [+0.41, +0.68] |
| (35, 55] | 395 | 98 | +0.143 [−0.21, +0.41] | +0.169 [−0.15, +0.45] |
| (55, 105] | 163 | 57 | **+0.282** [+0.03, +0.49] | **+0.287** [+0.04, +0.51] |
| > 105 ms | 167 | 62 | **−0.542** [−2.07, −0.08] | +0.266 [−0.00, +0.46] |
| **all > 15 ms** | 4,496 | 272 | **+0.191** [+0.10, +0.28] | **+0.388** [+0.32, +0.46] |
| **all > 45 ms** | 435 | 105 | −0.057 [−0.45, +0.21] | **+0.271** [+0.12, +0.41] |

| Δt | n | shots | `CES_VT` vs PCHIP | `CES_VT` vs **persistence** |
|---|---:|---:|---|---|
| ≤ 15 ms | 51,457 | 195 | +0.209 [−0.02, +0.29] | **+0.368** [+0.17, +0.45] |
| all > 15 ms | 1,756 | 161 | +0.027 [−0.20, +0.24] | **+0.309** [+0.13, +0.47] |
| all > 45 ms | 27 | 11 | **−0.920** [−2.89, −0.50] | +0.126 [−0.40, +0.43] |

**Three findings, in decreasing order of confidence.**

1. **The `CES_TI` win is NOT confined to adjacent history — this is new.** Pooling makes the
   non-adjacent regime measurable for the first time, and the model beats future-using PCHIP there:
   **+0.191 [+0.10, +0.28] over all Δt > 15 ms**, with (15,25], (25,35] and (55,105] each
   significant on their own. §4.3's "the advantage concentrates in the small-gap regime" understated
   the result — it was a power problem, not a skill problem.
2. **At Δt > 45 ms the "collapse" does not survive contact with a CI.** vs PCHIP it is −0.057
   [−0.45, +0.21], i.e. **n.s.**; the single-seed −8.98 was noise from n = 45. And vs the causal
   baseline the model **wins significantly there (+0.271 [+0.12, +0.41])**. Only in the extreme
   > 105 ms bin does PCHIP significantly beat the model (−0.542, CI entirely negative), where it is
   interpolating between two genuine anchors across a gap the model must extrapolate.
   Correct framing: **the wide-gap regime belongs to two-sided interpolation, which is exactly what
   an online system does not have.**
3. **`CES_VT` ties interpolation but beats every causal alternative at every resolvable gap**
   (+0.368 small, +0.309 beyond 15 ms, both PASS). The headline "`V_rot` is n.s." is a statement
   about *future-using* methods only. For an online virtual sensor the rotation channel is useful.

**Caveat retained.** The `stuck0` arm of the same analysis blows up in the widest `CES_VT` bin
(−156 vs PCHIP, n = 46) precisely because held targets have ≈0 baseline error — an independent
demonstration that hold-inclusive scoring is not usable at large Δt. Artifacts:
`data/.largegap_rerun.json`, `data/.largegap_analysis.json`.

---

## 8h. Paper/artifact reconciliation (2026-08-05) — the draft did not match the repo

An audit of `docs/paper/main.tex` against the frozen artifacts found four defects. All are fixed;
`ces_prediction/collect_paper_numbers.py` now regenerates `docs/paper/paper_numbers.json` from the
run directories, and `docs/paper/make_figures_en.py` reads that file instead of hard-coded literals,
so text/table/figure drift cannot recur.

1. **§Model described the wrong architecture.** It documented `ces_prediction/model.py` (2-layer
   pre-LayerNorm Transformer, learned positional embeddings, attention-weighted mean *and standard
   deviation*, 816,556 params) — a model that produced **no** number in the paper and **cannot load
   any saved checkpoint**. The published model is `model_iter009.py`: 1-layer **bidirectional GRU**
   (hidden 64) + per-target multi-head additive attention **hard-masked to observed timesteps** +
   per-target routing, **201,258 params**. The observation mask — iter009's actual defining idea and
   a clean statement of *why* it works (inject interpolation's inductive bias at zero parameter
   cost) — **was missing from the paper entirely** and is now a stated contribution. The §peak
   mechanism story rested on the non-existent attention-weighted std and was removed.
2. **Headline numbers came from a superseded checkpoint family.** The paper quoted `CES_TI`
   +0.270/+0.200/+0.271/+0.295 and genuine +0.225/+0.204/+0.285/+0.288; the reproducible family on
   disk gives **+0.2566/+0.1937/+0.2634/+0.2796** (hold-inclusive) and **+0.1792/+0.1971/+0.2802/
   +0.2628** (genuine), matching §4.1. Ablation (+0.36/+0.39/+0.16, −3.31) and peak
   (+0.52→+0.86, +0.24→+0.69) were stale the same way; the artifacts give **+0.43/+0.46/+0.37,
   −0.64** and **+0.27→+0.70, +0.13→+0.44**. Every conclusion survives; only magnitudes moved.
3. **The evaluation population was mis-stated.** The paper claimed all conclusions used genuine-only
   evaluation while quoting the hold-inclusive population (`CES_VT` n = 27,437). The genuine
   population is now the headline (seed 42: n = 32,787 `CES_TI` / 96 shots, **10,729 `CES_VT` / 61
   shots**) with hold-inclusive kept as an explicit sensitivity column. Two consequences must be
   stated rather than buried: genuine `CES_TI` vs the **stronger linear** interpolant is **3/4 PASS**
   (seed 42 n.s.), and genuine `CES_VT` is **1/4 PASS** (seed 1), not 0/4 — one in four is what noise
   gives, so the tie verdict stands, but "not significant on any split" was wrong.
4. **§1 (held values) contradicted §8c.** The draft still said held values do not contaminate
   training. Replaced with the 4-seed paired result and the §8f-R monotone-decay mechanism.

---

## 8i. MNAR quantified (2026-08-05) — the causal claim survives, the offline claim does not

**Motivation.** "Skill is measured on observed points only; CES missingness is MNAR, so this is an
optimistic bound" was the paper's biggest un-numbered caveat — an unbounded correction attached to
the headline. This closes it.

**Method** (`ces_prediction/experiments/mnar/analyze_mnar.py`; evaluation only, no retraining).
Post-stratify on two covariates computable at **observed and missing rows alike**, then reweight the
scored points onto the missing rows' distribution:
- `dt` = time since the last *genuine* observation of that target (held values NaN'd exactly as the
  genuine dataset does, via `KSTAR_CES_Dataset._stuck_repeat_mask`);
- `act` = the input-only local-activity flag of `peak_analysis.detect_peak_rows_input_only`, which is
  defined at a missing row because its neighbour set **excludes the row itself**.

Guards, each load-bearing:
- **Domain.** `compare_baselines` scores a sample only if an observed value lies *inside* the window,
  so a missing row whose nearest past observation is further back than `W−1` rows is not hard — the
  model **cannot be applied to it**. Counting those as fillable would invent a population.
- **Sufficiency.** Strata with < 30 scored samples are dropped, not trusted, and the covered mass is
  reported next to every estimate. Without this the `CES_VT` estimate was driven by a stratum holding
  **15,404 missing rows and 6 scored samples**.
- **Alignment.** The scored side takes `act` from the frozen npz, the weights from a grid that also
  keeps all-missing rows, so thresholds could diverge. Measured: activity rate 0.124 vs 0.128
  (`CES_TI`) and 0.233 vs 0.241 (`CES_VT`) — same variable.
- CI = shot-clustered bootstrap (B = 2,000) on the scored side at fixed weights; covers error
  variability, not weight-estimation variability (stated).

**Result 1 — a scope fact not previously stated.** Of genuinely missing rows, the fraction that is
**in-domain** at `W = 4` (an observed value inside the window) is **54.1% for `CES_TI`** and
**4.8% for `CES_VT`**. At this window the method addresses about half the real `T_i` gap problem and
a corner of the `V_rot` one. This is a *coverage* limit, not an accuracy one, and it re-opens §8f:
window size is irrelevant to skill but decisive for how much of the real missingness is reachable.
It is the strongest argument yet for decoupling history **reach** from history **depth**.

**Result 2 — skill reweighted onto the in-domain missing population** (95–100% stratum coverage):

| target / baseline | seed 42 | seed 1 | seed 7 | seed 123 | optimism |
|---|---|---|---|---|---|
| `CES_TI` vs **persistence** | **+0.292** [+0.162,+0.415] | **+0.308** [+0.143,+0.429] | **+0.293** [+0.192,+0.441] | **+0.293** [+0.155,+0.392] | +0.06…+0.12 |
| `CES_TI` vs PCHIP | +0.061 [−0.075,+0.267] | +0.132 [−0.064,+0.274] | **+0.211** [+0.050,+0.374] | +0.175 [−0.025,+0.290] | +0.06…+0.09 |
| `CES_VT` vs **persistence** | +0.139 n.s. | **+0.415** [+0.237,+0.584] | +0.364 n.s. | **+0.269** [+0.178,+0.491] | −0.05…+0.21 |
| `CES_VT` vs PCHIP | −0.014 n.s. | **+0.189** [+0.024,+0.396] | +0.221 n.s. | −0.230 n.s. | −0.10…+0.41 |

**Interpretation — the two comparisons come apart, and that split is the finding.**
- Against **persistence**, the `CES_TI` advantage survives reweighting **4/4 with a strikingly stable
  +0.29**, and the MNAR correction costs only 0.06–0.12. This is the deployable claim, and it is now
  measured at the missing points rather than extrapolated to them.
- Against **future-using PCHIP** it does **not** survive (1/4). Mechanically consistent with §8g:
  missing points sit at larger `dt`, exactly where a two-sided anchor helps most and a causal
  extrapolator least.
- Honest consequence: **"beats offline interpolation" is a statement about the observed population**
  (where it is the harder, more informative bar and should stay the headline); **"beats every causal
  method" is the statement that holds at the points a nowcaster actually fills.** Conflating them is
  the main way this result could be oversold.
- `CES_VT` behaves the same way on a far thinner base (4.8% in-domain), so no deployment conclusion
  is drawn for rotation.

Artifact: `data/.mnar_analysis.json`.

---

## 8j. Framing correction (2026-08-05, 승상님) — "the information is poor" is not a conclusion

Recorded because it changed the paper, not just the wording. The draft had drifted toward
"information-limited, not capacity-limited" as a *terminal* claim, which reads as an excuse and
gives a reader nothing to do. The accumulated negatives (§8e CT encoders, §8f window sweep, MC
derived features, seq reframing) are worth stating **only** as routing information: they say where
*not* to spend, which is what licenses a specific claim about where the next gain is. §Headroom of
the paper now names three levers, each grounded in this repo's own measurements:

1. **Reach, not depth.** 54.1% / 4.8% in-domain (§8i) is the binding limit on applicability, and
   §8f shows W buys coverage (dt > 15 ms: 456 → 1,958; dt > 45 ms: 14 → 135, W = 2 → 8) while
   buying no skill. Keep 2–3 history *slots*, draw them from a wider *span*
   (`dataset.py`'s declared-but-unused `max_window_span` is exactly this hook).
2. **The Mirnov information was destroyed by preprocessing, not missing from the plasma.**
   §8b.2: lag-1 autocorrelation is +0.568 (BES) / +0.572 (ECEI) / **−0.009 (MC)** on the *same*
   grid, 82% of MC blocks below |r| = 0.1. That is decimation of a kHz dB/dt signal to 100 Hz with
   no anti-aliasing filter — the mode-rotation frequency, the one plausible rotation proxy in the
   set, is gone before the model sees it. Fix is upstream: per-window RMS / band power / mode number
   from the **raw** kHz stream. Highest-value named experiment for `CES_VT`; testable on archived data.
3. **The actuator channel is absent.** §8b.3: `T_e`~`CES_TI` r = +0.353 (p = 3e−17) but
   `T_e`~`CES_VT` r = +0.024 (p = 0.58) — power is not torque. NBI torque is a data-acquisition task.

Rule going forward: **never report a negative result without naming the measurement that would
overturn it.**

---

## 8k. Complexity ladder (2026-08-05) — what a fully-explainable model actually buys

**Motivation (외부 피드백).** "모델이 너무 복잡하고 설명성이 부족하다." The wrong answer is to
show a tiny model that also works; the right answer is to **measure what the complexity earns**.

**Design** (`ces_prediction/experiments/anchor/`; 4-seed pre-registered batch). The anchor+Δ model
predicts, per target, a sum of three named terms:

```
ŷ = [(1−s)·y₁ + s·ȳ_obs]         anchor: nearest observed CES, learned smoothing s = σ(α)
  + β · m · g                     slope of the last two observations × gap
  + Σ_mod rate_mod(stats_mod)·g   per-modality (BES/ECEI/MC) rate × gap
```

**1,258 parameters** (0.6% of iter009's 201,258). Every learned term is zero-initialised (α = −4), so
training *starts exactly at persistence* and the skill is literally what learning added; `decompose()`
returns the per-term contribution, so any transient shot can be explained term by term.

Protocol fix applied before running: the runner had **not pinned `CES_DROP_STUCK_TARGETS`** — the
§8f trap. It now pins held-free at train *and* eval, pins the file split to the control manifest with
a leak check, and pairs against the **held-free** control `.sf_iter009_s*` (not `.vt_repro_*`), so the
paired difference isolates architecture rather than mixing in data treatment.

**Result — the ladder** (held-out TEST, `skill_vs_pchip`, held-free, genuine eval):

| arm | params | s42 | s1 | s7 | s123 | mean |
|---|---:|---:|---:|---:|---:|---:|
| persistence | 0 | −0.260 | −0.320 | −0.251 | −0.256 | **−0.272** |
| **anchor+Δ** | **1,258** | −0.107 | −0.138 | −0.097 | −0.109 | **−0.113** |
| iter009 (published) | 201,258 | +0.238 | +0.184 | +0.291 | +0.221 | **+0.234** |

Fraction of the persistence → full-model gap recovered by the anchor: **30.7 / 36.1 / 28.4 / 30.8 %**
(mean **31.5%**, `CES_TI`) and **5.9 / 5.6 / 10.7 / 5.6 %** (mean **7%**, `CES_VT`).
Paired verdict: **FAIL** (mean paired `CES_TI` −0.455, 0/4 favor, 4/4 against, pchip PASS 0/4).

**Interpretation — this is the useful answer to the complexity criticism.**
1. **The complexity is earning its keep, and we can now say by how much.** A fully interpretable
   model with 0.6% of the parameters recovers ~31% of the `T_i` margin and ~7% of `V_rot`'s. The
   remaining 69% / 93% is what the learned multimodal encoder buys.
2. **The anchor is still far above persistence** (−0.113 vs −0.272), so even the named-terms form
   learns something real — it is a legitimate rung, not a strawman.
3. **The gap is informative about the problem, not just the model.** The anchor is *linear in the
   gap* (anchor + slope×gap + rate×gap). That it captures only a third says the recoverable structure
   is substantially non-linear/non-local — so a more explainable model needs a different functional
   form, not more linear terms. The `CES_VT` figure (7%) says its signal is not a local slope at all.
4. Consistency is the strongest feature: 28–36% across four independent splits.

Artifacts: `data/.anchor_s{42,1,7,123}`, `data/.anchor_summary.json`.

---

## 8l. Real-time feasibility measured (2026-08-05) — and the GPU is the wrong device

`ces_prediction/experiments/latency/bench_latency.py`. Network forward pass only (feature assembly
and normalization happen outside the model); RTX 5060 Laptop + torch 2.11.0+cu128, unloaded machine,
1,000 iterations after 50 warmup, **p95/p99 reported because a control loop is decided by its tail**.

| W | device | batch | per-call median | p99 | amortized | throughput |
|---|---|---:|---:|---:|---:|---:|
| 2 | **CPU** | 1 | **2.76 ms** | **8.74 ms** | 2.60 ms | 385/s |
| 4 | **CPU** | 1 | **2.83 ms** | **6.36 ms** | 3.16 ms | 317/s |
| 2 | CUDA | 1 | 21.70 ms | 42.60 ms | 22.97 ms | 44/s |
| 4 | CUDA | 1 | 21.45 ms | 72.01 ms | 28.83 ms | 35/s |
| 2 | CPU | 512 | 10.61 ms | 24.10 ms | 10.58 ms | **48,390/s** |
| 2 | CUDA | 512 | 18.96 ms | 43.63 ms | 21.07 ms | 24,305/s |

1. **Online inference fits the 10 ms CES budget on CPU**: p99 = 6.4–8.7 ms, i.e. 64–87% of one grid
   period, with the median at ~2.8 ms. The claim "real-time" is now measured, not asserted.
2. **Do NOT deploy this on a GPU for single-sample inference.** CUDA batch-1 is ~8× slower than CPU
   and blows the budget (p99 4.3–7.2× over). At 0.2M parameters there is nothing to amortize the
   launch overhead against. This was verified two ways — per-call timing (which forces device idle,
   the realistic model for a 10 ms loop) and amortized back-to-back timing — and both agree, so it
   is not an artifact of forcing the GPU idle between calls.
3. CPU also wins on bulk reprocessing here (48k vs 24k samples/s at batch 512). Likely specific to a
   laptop GPU and a very small model; stated as measured on this hardware, not as a general claim.
4. Run-to-run p99 varies (6.2 → 8.7 ms across two runs of the same configuration), so the honest
   statement is "within the budget with limited headroom on a laptop CPU"; a dedicated control
   machine would have more.

Artifact: `data/.latency_benchmark.json`.

---

## 8m. Calibrated predictive intervals (2026-08-05) — split conformal, no retraining

**Motivation.** A virtual sensor with no error bar is unusable, and the fusion community's default
tool (GP regression) returns a posterior while ours returns a point. A learned variance/quantile head
would require retraining, which would move the point predictions and confound every skill number in
the paper. **Split conformal** avoids that entirely: it calibrates on a held-out split, changes
nothing about the predictor, and gives distribution-free finite-sample marginal coverage.

**Design** (`ces_prediction/experiments/uq/`). Score = |residual| = √(se). Calibrate on the VAL split,
evaluate on TEST, both genuine treatment (val artifacts regenerated by `rerun_compare_val.py`, since
the only val npz on disk was held-kept — calibrating on one population and testing on another would
silently void the guarantee). Two variants: `global` (one quantile) and `mondrian` (a quantile per
dt × activity stratum, falling back to global below 50 calibration points). The **identical**
procedure is applied to PCHIP and persistence, so differences are interval quality, not calibration
technique. α = 0.10.

**Result (mean over 4 seeds; Winkler interval score, lower = better):**

| target | mode | coverage | Winkler: model / pchip / persistence | width vs pers. | Winkler vs pers. |
|---|---|---|---|---:|---:|
| `CES_TI` | global | 87.1–91.7% | model best on 4/4 | 0.966 | **0.807** |
| `CES_TI` | mondrian | 87.0–91.4% | model best on 4/4 | **0.884** | **0.796** |
| `CES_VT` | global | 90.8–93.8% | model best on 4/4 | 0.944 | **0.827** |
| `CES_VT` | mondrian | 91.0–94.0% | model best on 4/4 | 0.937 | **0.831** |

1. **The model's intervals are strictly better than both baselines' by a proper scoring rule, on
   every seed, target and mode (8/8 each).** Winkler ≈ 0.80 of persistence's. Note this is not
   implied by better point estimates — an interval score penalises width and misses jointly.
2. **Mondrian beats global everywhere** (e.g. `CES_TI` s42: 2013.7 vs 2207.0) — putting width where
   dt and activity say it is needed pays, even though the *mean* width rises.
3. **The information asymmetry shows up in the uncertainty too, weakly.** At matched coverage the
   model shrinks the interval to 0.884 of persistence's for `CES_TI` but only 0.937 for `CES_VT` —
   same direction as the point-estimate asymmetry, much smaller in magnitude. Honest reading: it
   partially "knows what it does not know", and the effect is weak because persistence is already
   a strong `V_rot` predictor.
4. **Honest failure: coverage is marginal, not conditional.** `CES_TI` under-covers on seeds 1 and
   123 (87.0–88.9% against a 90% target) and per-shot coverage ranges from ~50–68% at the 10th
   percentile to 100% at the 90th. Conformal guarantees exchangeability-based *marginal* coverage;
   calibration and test are disjoint **shots**, and shot-level distribution shift breaks that. For a
   deployed instrument this matters more than the pooled number: on some discharges the interval
   is systematically too narrow. Fixing it needs shot-conditional (or Mondrian-by-shot-cluster)
   calibration, which the current shot count does not support.

Artifacts: `data/.uq_conformal.json`, `data/.uq_val_rerun.json`, `<run>/comparison_errors_val__val_genuine.npz`.

---

## 8n. Campaign shift (2026-08-05) — the offline claim dies, the causal claim survives, and we know why

**Motivation.** Every result up to here rests on a *seeded random* file-level split. That is the
right control for row-adjacency leakage, but it is not what deployment faces: a deployed nowcaster
trains on the shots that exist and runs on shots that do not exist yet. A random split hands the
model future discharges to learn from.

**Design** (`ces_prediction/experiments/campaign/run_campaign_shift.py`). Order all 641 discharges by
shot number (30801–32751, contiguous, no campaign gap > 200) and cut strictly in time:
**train 416 shots [30801, 31991] → val 128 [32002, 32310] → test 97 [32312, 32751]**. No test shot
precedes any train shot (asserted). Everything else pinned to the headline protocol (iter009, W = 4,
held-free, same caps). The split is FIXED, so the four runs vary the **init seed only** — replication
is over initialisation, not over splits; the shot-clustered CI inside each run still covers
shot-to-shot generalisation. 97 test shots ≈ the random split's 96, so power is comparable.

**Result (held-out TEST, 4 init seeds):**

Full PR4 treatment against **both** bars (`bootstrap_vs_persistence.py` — `bootstrap_compare.py`
only gates against the interpolants, so the claim that actually survives had been left as a bare
point estimate):

| seed | `CES_TI` vs PCHIP | `CES_TI` vs **persistence** | `CES_VT` vs PCHIP | `CES_VT` vs **persistence** |
|---|---|---|---|---|
| 42  | +0.051 [−0.086, +0.135] n.s. | **+0.275 [+0.175, +0.342] PASS** | −0.061 n.s. | **+0.223 [+0.016, +0.316] PASS** |
| 1   | +0.044 [−0.116, +0.145] n.s. | **+0.270 [+0.153, +0.343] PASS** | −0.005 n.s. | **+0.264 [+0.099, +0.356] PASS** |
| 7   | −0.148 [−0.395, −0.000] n.s. | +0.123 [−0.061, +0.241] n.s. | −0.156 n.s. | **+0.153 [+0.019, +0.242] PASS** |
| 123 | −0.018 [−0.195, +0.094] n.s. | **+0.222 [+0.099, +0.303] PASS** | −0.056 n.s. | **+0.226 [+0.025, +0.355] PASS** |
| **verdict** | **0/4 PASS** (linear: 0/4) | **3/4 PASS**, mean +0.222 | **0/4** (linear: 0/4) | **4/4 PASS**, mean +0.216 |

Note the `CES_VT` row: the target that "ties interpolation" globally is **significantly better than
persistence on 4/4 initialisations even across a campaign boundary**. Rotation is not a failed
target; it is a target whose only useful comparison is the causal one.

The test period is genuinely harder for everyone (PCHIP RMSE 601 vs 449 on the random split;
persistence 688 vs 504), but skill normalises that away, so the drop is a real loss **relative to
interpolation**, not just harder data. All four runs share identical baselines (the split is fixed),
which confirms the design.

**This is the second independent stress test, and it agrees with the first (§8i).**

| stress test | vs PCHIP (offline) | vs persistence (causal) |
|---|---|---|
| random split, observed points | **+0.18…+0.28, 4/4 PASS** | +0.35…+0.42 |
| reweighted to genuinely missing points (§8i) | +0.06…+0.21, **1/4** | **+0.29, 4/4 PASS** |
| temporal campaign split (§8n) | −0.15…+0.05, **0/4** | **+0.12…+0.28** |

The pattern is unambiguous: **the advantage over future-using interpolation is a property of the
observed population of a random split and survives neither stress test; the advantage over the
baselines a real-time system can actually run survives both.** That is the claim the thesis should
make about deployment, and it is now supported by two independent robustness analyses rather than
asserted.

### Why it degrades — measured, not asserted

`ces_prediction/experiments/campaign/diagnose_shift.py`. Per-channel drift from the train period to
the test period, in the units the model was normalized in (`z = |mean_test − mean_train| / std_train`):

| group | channels | z median | z max | std ratio (median) |
|---|---:|---:|---:|---:|
| **BES** | 9 | **1.218** | 1.304 | 0.750 |
| **ECEI** | 4 | **0.529** | 0.600 | 0.620 |
| MC | 2 | 0.007 | 0.009 | 0.724 |
| **`CES` (targets)** | 2 | **0.115** | 0.147 | 1.058 |

**The fast diagnostics drift 5–11× more than the target does.** BES moves 1.22 σ in location and
loses 25% of its scale; ECEI moves 0.53 σ and loses 38%; the CES targets barely move at all
(0.115 σ, scale ratio 1.06). MC shows no location drift because it is zero-mean noise (§8b.2) but
its scale shrinks 28%.

That is exactly the mechanism the skill pattern implies. The pathway that gives the model its edge
*over CES-only interpolation* is the fast-diagnostic pathway, and that is the one fed inputs 1.2 σ
outside the range its encoder was normalized on. The CES-history pathway sees the same physical
quantity in the same units in both periods, which is why the margin over persistence largely holds.

**Named fix (per the §8j rule).** The pipeline normalizes with **train-file-only statistics** — the
correct leakage-hardening choice for a random split, and precisely what breaks under campaign shift.
The targeted repair is **per-shot (or causal running) standardization of the fast diagnostics**,
which uses only that shot's own data and is therefore available at run time and still leak-free.
Caveat worth an experiment rather than an assumption: per-shot standardization also removes
absolute-level information, which may itself carry `T_i` signal (higher BES ↔ higher density), so it
could cost skill on the random split while buying transfer. That trade-off is the next controlled
experiment, and it is cheap: one env-level change to the normalization, same 4-seed protocol, scored
on both split rules.

Artifacts: `data/.campaign_s{42,1,7,123}`, `data/.campaign_summary.json`,
`data/.campaign_split_manifest.json`, `data/.campaign_shift_diagnosis.json`.

---

## 8o. Full defense audit + literature re-verification (2026-08-05) — no experiment, documentation loop

An 11-axis audit (claims / novelty / related-work benefit / dataset+training / preprocessing /
statistics / baselines / interpretability / conclusion consistency / applications / anticipated
questions) of every claim-bearing artifact. Findings and actions:

1. **`docs/연구설명_목차별정리.md` was stale** — still said held values do not contaminate
   training (overturned by §8c), described the wrong architecture (the Transformer `model.py`,
   §8h defect #1), quoted the hold-inclusive headline, and omitted every 2026-08-05 experiment.
   **Rewritten in full** against §8c–§8n + `paper_numbers.json`.
2. **All four existing decks (1h/20min/flow/1-pager) lagged the paper** — hold-inclusive headline,
   the overturned "held does not contaminate training" claim, no stress tests / ladder / conformal /
   latency. **Rebuilt same day**: `make_figures.py` now reads `paper_numbers.json` (no hard-coded
   run numbers), the 1-hour deck gained two slides (stress tests, deployment) and a §8g pooled gap
   table (40 slides), the 20-min deck's audit/limits slides and speaker notes were corrected, the
   flow deck's status/next-work slides were brought to post-§8g/§8n state, and the 1-pager's stale
   numbers (+0.86/+0.69 peak, +0.16/−3.31 ablation, Transformer description) were replaced — plus
   its long-standing missing-glyph tofu (ᵢ/ₑ/−/≈ absent from Malgun Gothic in matplotlib) fixed via
   mathtext. All decks preview-QC clean. The new audit deck `KSTAR_CES_종합방어.pptx`
   (`build_pptx_defense.py`, 20 slides) carries the novelty verification and Q&A defense table.
3. **Adversarial literature re-verification (2nd pass, through Aug 2026; 18 queries + 12
   primary-source fetches): all three novelty claims STAND.** Appended a dated section to
   `docs/paper/litreview/NOVELTY.md`. Consequences applied to the paper (both languages, static
   check clean, 51 cites resolved): Shousha year fixed 2023→2024; `Kim2024KineticProfile`
   (NF 64, 106052 — KSTAR CES `T_i` GP profile fitting, spatial not temporal) added to the GP
   paragraph; `Char2024FullShot` (arXiv:2404.12416 — rotation IS predicted when NBI actuators are
   inputs) added to §Headroom lever 3 as the positive control that makes the torque claim
   falsifiable; Jung2026 clause added ("consumes CES as an input, names rotation as future work").
   RTCAKENN is the closest threat; the three defenses (not fluctuation diagnostics / no
   past-target-history channel / never benchmarked vs interpolation) are now written down.
4. **Numeric cross-check of paper vs THESIS_RESULTS vs `paper_numbers.json`: no mismatches.**
5. New deliverables: `docs/연구방어_종합문서.md` (the 11-axis defense document incl. a 17-question
   Q&A table with defense ratings) + the deck above. `python -m pytest -q`: 33 passed.

Remaining (next loop candidates): per-shot standardization controlled experiment (§8n's named
repair). The other two candidates were executed the same day: GP baseline arm → §8p; CES
fit-failure sensitivity → §8q.

---

## 8p. GP baseline arm (2026-08-05) — the strongest offline arm, and the model TIES it

**Motivation.** `compare_baselines.py` has carried a GP arm since the harness was written
(`if B._HAVE_SKLEARN: methods.append("gp")`), but sklearn was never installed, so the fusion
community's standard profile-fitting method (Chilenski 2015; Michoski 2024) was silently skipped
and its absence defended only rhetorically (defense doc Q9). This closes it — evaluation only,
no retraining.

**Implementation** (`baselines_interpolation.predict_gp`, first actually-run version;
`experiments/gp/`). The never-run sklearn draft (full-block fit + per-sample L-BFGS) benchmarks
at 38 ms/fit → 42 min/seed: infeasible. Replaced with an **exact numpy/scipy GP**: Matern-3/2 +
white noise, **local fit on the nearest 16 past + 16 future observed neighbors** (with ≤ 80 ms
length scales the posterior is numerically insensitive to farther points), values standardized
within the neighbor set, hyperparameters selected **per sample by exact log marginal likelihood
over a fixed grid** (ls ∈ {5,10,20,40,80} ms × noise ∈ {1e-4..1e-1}) — fully deterministic,
0.94 ms/fit. No-leakage neighbor set and the PR2 persistence fallback are unchanged from the
draft. Crucially its NaN condition (no in-block past observation) is identical to `ar_local`'s,
so **adding the arm cannot shrink the scored population** — verified: all 4 genuine re-runs
reproduce the §8g reference npz **bit-for-bit (14 keys) + 4 new keys** (`se_gp`, `y_true` × 2
targets; `data/.gp_rerun.json`). Cross-check: recomputed model-vs-PCHIP skills match the headline
to 4 decimals (0.1792/0.1971/0.2802/0.2628).

**Result (genuine, held-out TEST, B = 10,000, shot-clustered):**

| seed | RMSE: GP / PCHIP / model (`CES_TI`) | GP vs PCHIP | **model vs GP** |
|---|---|---|---|
| 42  | 391.1 / 449.3 / 407.0 | **+0.242** PASS | −0.083 [−0.248, +0.035] n.s. |
| 1   | 442.4 / 499.7 / 447.8 | **+0.216** PASS | −0.024 [−0.130, +0.050] n.s. |
| 7   | 490.2 / 550.0 / 466.6 | **+0.206** PASS | **+0.094 [+0.014, +0.153] PASS** |
| 123 | 532.6 / 626.7 / 538.1 | **+0.278** PASS | −0.021 [−0.141, +0.053] n.s. |

`CES_VT`: GP vs PCHIP +0.18…+0.23 (3/4 PASS); model vs GP all n.s. (−0.10…−0.03).

**Findings.**
1. **GP is decisively the strongest offline arm** — +0.21…+0.28 over the pre-registered headline
   PCHIP on 4/4 `CES_TI` splits. Mechanism consistent with §8b.1's linear-beats-PCHIP: PCHIP
   passes exactly through noisy anchors; the noise-aware GP averages the noise out, going one
   step further than linear.
2. **The model TIES the strongest offline smoother on `CES_TI`**: significantly better on 1/4
   (seed 7 +0.094), significantly worse on 0/4, n.s. with small negative points on 3/4
   (mean ≈ −0.01). Subsets do not break the tie: peak 1/4 PASS (−0.08…+0.45), Δt ≤ 15 ms 1/4,
   Δt > 15 ms 0/4 (points −0.16…+0.05, n.s. — the future anchor helps GP most there, as expected).
3. **What this changes and what it does not.** The pre-registered claim (vs PCHIP, PR1; linear
   cross-check) stands — that is what pre-registration is for. But "beats future-using
   interpolation" must not be generalized to "beats every offline method": the honest statement
   is now **"the causal model matches the strongest future-using, per-sample-ML-tuned smoother
   on the observed population, and beats the pre-registered interpolants"**. The deployment
   claim is untouched — GP reads future anchors and does not exist online (§8i/§8n causal
   superiority unaffected).
4. **Named measurements that could overturn the tie** (§8j rule): (a) more test shots — seed 7
   shows the effect is resolvable at current power; (b) a *causal* (past-only) GP forecaster arm,
   which belongs in the causal ladder next to `ar_local` and has not been run.

Artifacts: `data/.gp_analysis.json`, `comparison_errors_test__test_genuine_gp.npz` per run dir.

---

## 8q. CES fit-failure sensitivity (2026-08-05) — the headline was CONSERVATIVE

**Motivation.** §8's retained limitation: `CES_TI` > ~3 keV values are spectral-fit failures
(global p99 = 2,089 eV, max 14,984 eV) surviving in the evaluation population. Expected rebuttal:
"do the artifacts inflate the result?" Now measured (evaluation only, from the `y_true` key added
in §8p; `experiments/fitfail/analyze_fitfail.py`).

**Result (genuine, vs PCHIP, paired — same rows removed from all arms):**

| seed | full skill | ≤ 3,000 eV (dropped) | ≤ 2,089 eV (dropped) |
|---|---|---|---|
| 42  | +0.179 PASS | **+0.362** PASS (−123) | +0.364 [−0.003, +0.600] n.s. (−234) |
| 1   | +0.197 PASS | **+0.470** PASS (−142) | **+0.482** PASS (−287) |
| 7   | +0.280 PASS | **+0.589** PASS (−140) | **+0.615** PASS (−257) |
| 123 | +0.263 PASS | **+0.538** PASS (−205) | **+0.575** PASS (−425) |

**Finding — the artifacts DEFLATE the headline, not inflate it.** Dropping the 0.4–0.6% of rows
above 3 keV keeps 4/4 PASS and roughly **doubles** the skill (+0.18…+0.28 → +0.36…+0.59). The
unphysical spikes are unpredictable for every arm; their enormous squared errors are similar
across arms and drag the MSE ratio toward 1 (the same heavy-tail mechanism as §8's bootstrap
spread). The headline population keeps them (no post-hoc row surgery on a pre-registered
protocol), and can now be described as **conservative with respect to this artifact**. The
stricter p99 threshold gives 3/4 PASS (seed 42 marginal at [−0.003, +0.600]), so the "roughly
doubles" magnitude — not the PASS verdict — is threshold-sensitive at the strictest cut.

Artifact: `data/.fitfail_analysis.json`.

---

## 8r. peak × held crosstab (2026-08-09) — the hypothesis was backwards, and `CES_VT` decomposes cleanly

**Motivation.** Two facts about `CES_VT` had never been crossed. §1: 54% of *observed*
`CES_VT` values are held (forward-filled instrument repeats). Peak analysis: `CES_VT`
PASSes at high-activity peaks (+0.69) while the global result is n.s. A held row's correct
answer *is* the previous value, so PCHIP passes exactly through it and no causal model can
win there. `docs/ces_vt_proposals.md` proposal 1 predicted that peaks would be
*genuine-rich* ("activity means the diagnostic was actually updating"), which would make
the global n.s. partly an artifact of the scored population. Evaluation only, no retraining.

**Method** (`experiments/heldpeak/`). `compare_baselines.py` gained one additive key,
`{target}_row` (the target's row index inside its shot) — `_shot` alone cannot address a
row, so no per-row property computed from the raw CSV could ever be aligned to the SEs.
The four trusted runs were re-scored under the **held-kept (`stuck0`)** treatment, because
under the headline `genuine` treatment held values are already NaN'd out of scoring and the
held column would be empty by construction. The held flag is recomputed from the raw CSVs
with the loader's own rule (`dataset._stuck_repeat_mask`), never a second definition, and
the row alignment is *proved* per file by matching the `time` column value for value.
Cells are scored with the same shot-clustered paired bootstrap as everything else
(B = 10,000, seed 12345).

**Reproducibility note — the additive-key check had to be sharpened.** The §8g/§8i/§8p
pattern demands the re-run reproduce the reference npz bit-for-bit. It did not, and the
reason is worth recording: **`se_model` is not bit-reproducible across sessions on this
machine** (identical weights, identical population; per-sample relative drift median
3 × 10⁻⁴; RMSE 372.3162 → 372.3135, i.e. the published 4-decimal number is unchanged) —
a float32 CUDA forward pass, not a population change. Every key that actually *defines the
population* (`shot`, `row`-aligned `dt_ms`, `is_peak`, and the pchip/linear/persistence SEs)
**is** bit-identical on all four seeds. So the check now requires bit-identity of the
population keys, bounds `se_model` RMSE drift (< 0.01 physical units; observed ≤ 0.0027),
and the merged artifact **keeps the reference `se_model`** so no published number can move
underneath us.

**Result 1 — the hypothesis is not just unsupported, it is backwards.** Peaks are
*held-richer* than the bulk, on 4/4 seeds:

| seed | held% (all) | held% inside peaks | held% in bulk |
|---|---:|---:|---:|
| 42  | 60.8% | **68.3%** | 58.2% |
| 1   | 54.0% | **61.9%** | 51.4% |
| 7   | 54.0% | **70.6%** | 47.8% |
| 123 | 53.5% | **72.8%** | 45.8% |

Mechanism: the input-only activity detector flags large neighbour-bracket slope and high
local neighbour variance, and a forward-filled *staircase* — flat, flat, jump — looks exactly
like local activity. **The peak detector is partly detecting the instrument's hold pattern.**

**Result 2 — `CES_VT` decomposes into four cells with completely different behaviour**
(skill vs PCHIP; `vs persistence` in the last column; * = 95% CI excludes 0):

| cell | seed 42 | seed 1 | seed 7 | seed 123 | vs persistence |
|---|---:|---:|---:|---:|---|
| **peak · genuine** | +0.627 | **+0.551\*** | +0.594 | +0.613 | **+0.75…+0.82, 4/4 PASS** |
| bulk · genuine | −0.010 | −0.051 | −0.142 | +0.053 | −0.07…+0.14, 1/4 |
| peak · held | −3.32 | +0.75 | −1.95 | +0.61 | undefined (persistence is exact) |
| bulk · held | −353 | −48.3 | −85.9 | −411 | undefined |

1. **On genuinely measured, high-activity rotation the model is strong**: +0.55…+0.63 against
   future-using PCHIP (positive on 4/4, significant on 1/4 — the cell holds only ~2.2–2.8k rows
   over ~63 shots, so power, not sign, is the binding constraint), and **+0.75…+0.82 against the
   causal baseline on 4/4**.
2. **On the quiet bulk the model ties interpolation** (≈0 on all four) — consistent with §5.1's
   reading that interpolation is near-optimal on smooth stretches.
3. **On held rows the comparison is structurally unwinnable** (−48 to −411): PCHIP passes exactly
   through a value that is by construction the previous one. These are 46–58% of the bulk.

**Result 3 — removing held rows lifts `CES_VT` on 4/4 seeds, but does not make it significant.**
Restricting the scored population to genuine rows: +0.154 → +0.201, +0.109 → +0.126,
+0.065 → +0.098, +0.127 → +0.170. Every seed improves; none reaches PR4. So the held rows
**do** dilute the global number, but they are not the whole reason `CES_VT` is n.s.

**`CES_TI` is the clean control** (≈0% held by construction): peak +0.59…+0.68 **PASS 4/4**
vs PCHIP, bulk +0.03…+0.13 (0/4 vs PCHIP, **4/4 vs persistence**). The "edge concentrates in
high-variability neighbourhoods" finding therefore does **not** depend on the held artifact —
it reproduces on a target that has none.

**What this changes.** The honest `CES_VT` sentence is no longer "n.s. globally, PASS at peaks".
It is: *more than half of the observed rotation population is instrument holds, where no causal
method can beat an interpolant that reads them; on the quiet genuine bulk the model ties
interpolation; and on genuinely measured high-activity rotation it beats every causal baseline
on 4/4 splits and is positive against future-using PCHIP on 4/4.* The global n.s. is the average
of three different regimes, not a uniform verdict.

**Named measurements that would overturn this** (§8j rule):
1. **Recompute the peak definition from genuine neighbours only.** Result 1 shows the current
   detector partly keys on hold staircases; a held-free detector should sharpen the
   peak · genuine cell. Evaluation only — no retraining.
2. **More shots.** peak · genuine is positive on 4/4 at +0.55…+0.63 with only ~63 shots per split;
   this is the same power ceiling as Q3.
3. **Repeat the decomposition on the held-free training family** (`.sf_iter009_s*`, §8c) — train
   and evaluate both held-free — to test whether peak · genuine reaches PR4 when the model was
   never taught the hold pattern either.

Artifacts: `data/.peakheld_analysis.json`, `data/.heldpeak_rerun.json`,
`comparison_errors_test__test_stuck0_held.npz` per run dir.

---

## 8s. Per-shot input standardization (2026-08-09) — §8n's named repair works, and costs nothing measurable

**Motivation.** §8n reported that a strictly temporal (campaign) split destroys the advantage
over offline interpolation (`CES_TI` 0/4 PASS) and measured the cause rather than asserting it:
between campaigns the **fast diagnostics drift 1.22 σ while the quantity being predicted drifts
0.115 σ**, so the input distribution slides out from under a model trained on earlier discharges.
The repair §8n named — and the only named repair behind the generalization weakness (defense doc
Q10, ●●○) — is to standardize the fast diagnostics **within each discharge**. This runs it.

**Implementation.** `dataset.py` gained `CES_PER_SHOT_NORM` (default 0). When set, each shot's
BES/ECEI/MC channels are z-scored by **that shot's own** mean/σ at load time, so every downstream
stage (train-file-only global stats, disk cache, model inputs) sees one consistent treatment.
**Targets are never touched** — per-shot target scaling would make the physical-unit inverse
transform shot-dependent and break every baseline comparison. Verified directly: with the flag on,
per-file input mean/σ = 0.0000/1.0000 (off: 0.07/0.70, 0.00/0.70, 0.12/1.38, 0.12/2.34 on the first
five shots — the drift §8n measured), while the target column's nanmean is bit-identical either way.
The flag enters the dataset cache signature, so caches auto-invalidate.

Two arms, because a repair aimed at transfer must be checked where it was aimed **and** where the
paper's claim lives. Both pin every knob explicitly in the runner env (§8f).

### Part 1 — campaign split: the repair works (4/4 significant)

Control = §8n's own runs (`.campaign_s*`), same fixed temporal manifest, same caps, same held-free
treatment, same architecture; single controlled variable `CES_PER_SHOT_NORM`
(`experiments/campaign/run_campaign_shift.py --arm per_shot`, `summarize_pershot.py`).

| init seed | base (§8n) | per_shot | paired (per_shot − base) |
|---|---:|---:|---|
| 42  | +0.051 | **+0.183 PASS** | **+0.139 [+0.057, +0.237]** |
| 1   | +0.044 | +0.135 | **+0.096 [+0.007, +0.199]** |
| 7   | −0.148 | **+0.186 PASS** | **+0.291 [+0.160, +0.420]** |
| 123 | −0.018 | +0.079 | **+0.095 [+0.030, +0.170]** |
| **verdict** | 0/4 PASS | **2/4 PASS** | **4/4 favour, 4/4 CI excludes 0**, mean **+0.155** |

The control column reproduces §8n exactly (+0.051 / +0.044 / −0.148 / −0.018), confirming the
pairing. `CES_VT` is unmoved (1/4 favour, mean −0.019) — expected, and a useful negative control:
iter009's V_rot head is **blocked from the fast diagnostics by construction**, so no
standardization of those channels can reach it.

### Part 2 — headline split: no measurable cost

The arm cannot be adopted just for winning where it was aimed: per-shot standardization **discards
absolute level**, and the paper states explicitly that absolute level may itself carry `T_i`
information. Control = **`.sf_iter009_s*`** (the §8c held-free family) — *not* `.vt_repro_*`:
§8c established held-free training as the treatment every new run uses, and pairing against the
older held-kept family would have made the data treatment a second uncontrolled variable.
(`experiments/pershot/run_pershot_random.py`; the §8c split-leak guard is reused.)

| seed | per_shot skill | paired vs control (`CES_TI`) | paired (`CES_VT`) |
|---|---:|---|---|
| 42  | +0.141 | −0.127 [−0.330, +0.008] n.s. | +0.004 [+0.000, +0.009] |
| 1   | +0.162 PASS | −0.027 [−0.081, +0.011] n.s. | +0.001 n.s. |
| 7   | +0.269 PASS | −0.032 [−0.086, +0.009] n.s. | −0.002 n.s. |
| 123 | +0.254 PASS | +0.043 [−0.022, +0.129] n.s. | +0.003 n.s. |
| **verdict** | 3/4 PASS | **0/4 significant losses**, mean −0.036 | mean +0.002 |

**Verdict: ADOPT**, under the pre-registered rule (no seed shows a significant paired loss).
Stated honestly: the headline point estimates are **slightly negative on 3/4 seeds** (mean −0.036,
worst seed 42 at −0.127 with the CI upper bound only just above zero at +0.008), so this is "no
measurable cost", not "free". What is bought with it is large and significant: +0.155 mean under
campaign shift, where the base arm was **below** interpolation on two of four seeds.

**Why this matters beyond the number.** The transfer failure was the one weakness with no
demonstrated repair (Q10 ●●○, and §8n could only *name* the fix). It is now measured, and the
trade is quantified in both directions rather than assumed. It also confirms §8n's diagnosis
causally: if the campaign loss were about physics changing between campaigns, removing an input
*level* shift could not have recovered it.

**Named measurements that would overturn this** (§8j rule):
1. **Causal running standardization** instead of per-shot (an online estimator cannot see a shot's
   future to compute its σ). Per-shot is the offline upper bound of this family; the deployable
   version is an expanding-window or EWMA estimator, and the gap between them is unmeasured.
2. **More campaigns.** One temporal cut with a fixed split means replication here is over
   initialisation only (§8n's own caveat), so the +0.155 is a single-split effect measured four times.
3. **Per-channel diagnosis.** BES drifts 1.22 σ; whether the recovery comes from BES alone is
   testable by standardizing one modality at a time.

Artifacts: `data/.pershot_summary.json`, `data/.pershot_random_summary.json`,
`data/.campaign_ps_s*`, `data/.psr_s*`.

---

## 8t. seq v2 (2026-08-09) — the structural design, assembled: matches or beats the window pipeline at 1/10 the cost

**Motivation (승상님).** Ask what this problem is *structurally isomorphic to*, rather than
what pipeline we happen to have. It is not sparse-target regression; it is **latent state
estimation under multi-rate sensor fusion**: one plasma state, observed densely and quickly
(BES/ECEI/MC, 100 % of the grid) and sparsely and noisily (CES, 8.2 % / 23.9 % missing plus
41.1 % holds). That framing forces four design decisions — and this repository discovered
each of them separately, as its own controlled experiment, over eight months:

| what the isomorphism forces | where we found it instead |
|---|---|
| state exists at unobserved times → **full grid + loss-side masking** | §8d (`CES_TI` 4/4 improved) |
| a hold is **not an observation** (value ≠ observation model) | §8c (4/4 improved) |
| each discharge is an independent realisation → **per-shot standardization** | §8s (campaign 4/4 significant) |
| an observation function that is physically zero must be **structurally blocked** | iter009's V_rot routing |

Nobody had ever trained a model with all four. §8d closed by naming exactly this
("seq v2 = this framing + the iter009 V_rot routing, trained held-free"); this runs it, with
§8s's per-shot standardization added.

**Implementation** (`experiments/seq/model_seq_v2.py`, arm `--arm seq_v2`). Two causal LSTMs:
`T_i` reads the full 22-channel state; `V_rot` reads **only the non-fast tail** (dt plus each
target's carry-forward value, staleness and has-observation flag). The split is at the
**encoder**, not the head — a shared recurrent state would carry fast-diagnostic information
into the V_rot output no matter how the head is wired. Verified as a structural property, not
a hope: perturbing only the 15 fast channels changes `T_i` and leaves `V_rot` bit-identical.
357,570 parameters (v1: 344,898; budget < 1M). Control = `.sf_iter009_s*`, the §8c held-free
family — the only control that shares the data treatment.

**Result (held-out TEST, genuine population, 4 seeds).** Own skill vs PCHIP, one ladder:

`CES_TI`:

| seed | `sf_iter009` (control) | `+per_shot` (§8s) | **`seq_v2`** |
|---|---:|---:|---:|
| 42  | +0.238\* | +0.141   | **+0.255\*** |
| 1   | +0.184\* | +0.162\* | **+0.208\*** |
| 7   | +0.291\* | +0.269\* | **+0.305\*** |
| 123 | +0.221\* | +0.254\* | **+0.308\*** |

`CES_VT`:

| seed | `sf_iter009` (control) | `+per_shot` (§8s) | `seq_v2` |
|---|---:|---:|---:|
| 42  | +0.227   | +0.229   | +0.206 |
| 1   | +0.202\* | +0.203\* | +0.200\* |
| 7   | +0.143   | +0.141   | +0.110 |
| 123 | +0.210   | +0.212   | +0.200 |

(\* = own PR4 gate passed.) **`seq_v2` is the highest `CES_TI` on all four splits** and keeps
4/4 PASS; `CES_VT` is indistinguishable across the ladder (1/4 PASS everywhere).

Paired against the control (shot-clustered, same rows):

| | seed 42 | seed 1 | seed 7 | seed 123 | verdict |
|---|---|---|---|---|---|
| `CES_TI` | +0.021 | +0.029 | +0.019 | **+0.111\*** | **4/4 positive**, mean **+0.045**, 1/4 significant |
| `CES_VT` | −0.026 | −0.003 | −0.039 | −0.013 | 4/4 negative, **0/4 significant**, mean −0.020 |

**Finding 1 — the routing closes §8d's `V_rot` deficit.** §8d's third comparison (v1 held-free
vs the same control) lost `CES_VT` **significantly on 4/4 seeds** (−0.044, −0.072, −0.067,
−0.023, every CI excluding zero). With the encoder-level routing that becomes **0/4
significant** (−0.026, −0.003, −0.039, −0.013). This is attributable to the routing rather
than to per-shot standardization: in v2 the `V_rot` branch never sees the fast channels, so
standardizing them cannot reach it — which §8s independently confirmed on iter009, whose
`V_rot` head is blocked the same way.

**Finding 2 — the framing wins `CES_TI` again, now from a stronger control.** §8d measured
+0.034 mean against the held-kept control; v2 measures **+0.045 against the held-free control**
(4/4 positive), and unlike §8d it does not pay for it in `V_rot`.

**Finding 3 — it costs an order of magnitude less.** seq training is **1.2–1.4 min/seed**
against **12–22 min/seed** for the window pipeline. The window pipeline's cost is inherited
from one early decision — dropping rows with no observed target — which made the time axis
irregular and forced the window, the combinatorially explosive temporal-subset augmentation,
the sample caps, and eventually `CES_MAX_SAMPLES_PER_FILE` (§8f). The full-grid framing
deletes all of it.

**Finding 4 — the framing also dissolves a structural limit.** §8i showed only **54.1 %
(`CES_TI`) / 4.8 % (`CES_VT`)** of genuinely missing rows are even *in-domain* at W = 4. A
full-grid model has no in-domain/out-of-domain distinction: state is defined at every step.
This does not by itself make the missing points predictable, but it removes the framing's own
share of the restriction.

**Finding 5 — the two ingredients separate cleanly** (`--arm seq_v2_nops`: v2 routing with
per-shot standardization OFF, run immediately as §8t's own named follow-up, 5 min for 4 seeds).
Reading the `CES_VT` deficit down the ladder, all paired against the same held-free control:

| arm | routing | per-shot | `CES_VT` significantly worse |
|---|:--:|:--:|---|
| `seq_sf` (v1, §8d) | — | — | **4/4** (−0.044, −0.072, −0.067, −0.023) |
| `seq_v2_nops` | ✓ | — | **1/4** (−0.025, −0.009, **−0.049\***, −0.011) |
| `seq_v2` | ✓ | ✓ | **0/4** (−0.026, −0.003, −0.039, −0.013) |

**The routing does the work** (4/4 → 1/4); per-shot standardization closes the last seed
(1/4 → 0/4). On `CES_TI` the same ordering holds and is smaller: mean paired +0.034 (v1) →
+0.035 (routing) → +0.045 (both), with the sign going 3/4 → 3/4 → **4/4 positive**. So the
`V_rot` repair is the routing, and per-shot standardization contributes a consistent but small
`T_i` increment on top — which matches §8s, where its effect on the window pipeline's `T_i` was
also small (and slightly negative on the random split).

**What this does NOT show.** (a) Paired `CES_TI` significance is 1/4, not 4/4 — the sign is
consistent, the magnitude is small relative to seed noise, and the runner's built-in verdict
rule (designed for §8d, requiring 4/4 `a_better`) accordingly returns FAIL. The defensible
claim is **"matches or slightly beats, at a tenth of the cost"**, not "beats". (b) The two
pipelines do not share an identical training budget — seq uses val-based early stopping, the
window pipeline a fixed 10 epochs.

**Methodological reading (the actual point).** The isomorphism-first design and this project's
one-variable-at-a-time discipline are not competitors, they are an ordering. The controlled
experiments are what make each of the four ingredients *individually* credible — an
isomorphism-first model that bundled all four would never have shown that held values
contaminate training, or that per-shot standardization repairs campaign transfer specifically.
But the framing was available at the start, and it would have narrowed the search to
approximately where eight months of experiments arrived — at a tenth of the compute. The
practical lesson for the next problem: **derive the candidate design from the structure first,
then spend the controlled experiments proving its parts**, rather than starting from the
nearest supervised-regression template and recovering the structure through post-hoc
diagnostics.

**Named measurements that would overturn this** (§8j rule):
1. ~~Isolate the routing.~~ **Done** — see Finding 5.
2. **Equalise the training budget** (fixed epochs for both, or early stopping for both) before
   the +0.045 is quoted as an architecture effect.
3. **More seeds.** At 1.3 min/seed, 16 seeds costs 20 minutes and would settle whether the 4/4
   positive `CES_TI` sign is a real +0.045 or seed noise — the cheapest open question in the
   entire record.

Artifacts: `data/.seq_v2_lstm_s*`, `data/.seq_v2_nops_lstm_s*`, `data/.seq_v2_summary.json`, `data/.seq_v2_nops_summary.json`.

---

## 8u. Repository consolidation (2026-08-09) — the default import path is now the published model

No experiment; a wiring and scope pass, recorded because one of the changes removes a class of
silent error and another removes code that backed §8e.

**The wiring defect.** `ces_prediction/model.py` was the Transformer the retired AutoML search left
behind. It produced none of the published numbers and could not `load_state_dict` any saved
checkpoint — yet it was what `train.py` / `evaluate.py` / `compare_baselines.py` imported. Ten
runners worked around it by **copying `model_iter009.py` over `model.py` on disk and restoring it in
a `finally` block** (`swap_in_iter009` / `.wsweep_backup`). Two costs: a batch killed mid-run left
tracked source rewritten, and anyone running the pipeline without a runner silently scored the wrong
architecture.

**The fix.** `model.py` now re-exports `model_iter009.py`, and a variant architecture is selected
with **`CES_MODEL_FILE`** in the subprocess env like every other `CES_*` knob. Every swap/restore
helper is deleted. Verified: `model.MultimodalCESPredictor is model_iter009.MultimodalCESPredictor`;
`CES_MODEL_FILE=experiments/anchor/model_anchor.py` yields exactly **1,258 parameters** (§8k's
number); **10/10 historical checkpoints** (`.vt_repro_*`, `.sf_iter009_s*`) `load_state_dict`
against the default path, against 0/45 before; smoke train + `evaluate.py` run end to end with no
injection; 34/34 tests pass.

**Shared code promoted out of `ct/`.** `run_ct_experiments.py` had become the de-facto base module
for ten other batches (`SPLIT_SRC`, `BASELINE_OUT`, `FULL_ENV`, `run_step`), and
`paired_model_compare.py` lived under it too — so the CT batch could not be removed without taking
the rest with it. Both are now `experiments/runner_common.py` and
`experiments/paired_model_compare.py`.

**Scope removals** (git history keeps everything): `experiments/ct/` and the continuous-time claims
in `main.tex` / `main_ko.tex`; executed plan docs; literature-search intermediates (`NOVELTY.md`
kept, moved to `docs/paper/`); tracked `.omc/` scratch; LaTeX build artifacts. **§8e keeps the CT
verdict** — the record, not the code, is what stops a re-run at `W = 4`.

**Availability statements written.** The repository is public, so "the link will be inserted at
camera-ready" was replaced with the actual URL, and the two papers now carry separate **Code
availability** and **Data availability** sections. The data statement is deliberately narrow: the
641 shot files (30801–32751) are KSTAR experimental data governed by the operating institution's
policy and are *not* redistributed here, while the repo ships everything else needed to reproduce
the analysis once they are in place (split manifests, per-run env, per-sample squared-error
archives). Still open and **the user's calls, not ours**: target venue, co-authors/advisor, and an
archival DOI (a Zenodo mint needs their account) — the DOI is the one remaining `TODO(user)` in
either `.tex`.

**Papers rebuilt**: `main.pdf` 24 pp, `main_ko.pdf` 22 pp, both 0 errors / 0 unresolved references /
6 figures. The Korean build's three long-standing `Improper alphabetic constant` errors were
root-caused and fixed: `article.cls` typesets `\@author` inside a `tabular`, and a Hangul character
directly before `\\` breaks the row-break's optional-argument lookahead under xeCJK. Ending that
line with `{}` fixes it.

---

## 8v. Direction reset (2026-08-12) — full re-experiment under one confirmed protocol

No experiment; a decision record (taken 2026-08-12 on the Notion working page "KSTAR CES 연구",
transcribed into this ledger 2026-08-14). It governs every batch that follows.

**The decision.** Every quantitative number in the draft's §5–§7 — headline, MNAR reweighting,
campaign split, conformal intervals, interpretability ladder — was produced under the provisional
`W = 4` protocol. Three of this ledger's own results indict that protocol independently: §8f
(`W = 4` is not justified by skill; the plateau rule returns `W = 2`), §8c (held-free training wins
and is already the training default), §8q (the `CES_TI` fit-failure spikes roughly halve measured
skill). Rather than patching results one at a time onto mixed populations, **everything is re-run
once under a single confirmed protocol**; until a number is replaced it is provisional and is not
to be quoted:

> **Amendment (2026-08-14, 승상님):** the spike-*inclusive* ("p100") population is co-primary,
> not a sensitivity row — an unqualified claim must hold in BOTH the cut and the inclusive
> regime, each internally consistent across training/history/anchors/evaluation
> (`PREREGISTRATION_W2.md` §1.1; amended after partial unblinding of the w2cut window family,
> recorded there). ELM/transient-capture claims ride on the peak-stratified analyses, not on
> the population choice.

- **`W = 2`** — the plateau-minimal window (§8f).
- **held-free (`genuine`)** in training *and* evaluation (§8c).
- **`CES_TI` fit-failure spikes excluded by pre-registration**, consistently in all three places —
  training targets, history inputs (treated as missing), evaluation population — and identically
  for every arm; the spike-*inclusive* population is demoted to a sensitivity row. This inverts
  §8q's conservative choice legitimately because it is fixed **before** the re-run, not after. The
  cut threshold is re-justified from the current dataset's p99 with threshold sensitivity
  (2.5 / 3 / 4 keV) reported; if CES fit-quality metadata (fit χ², signal level) ever arrives, a
  quality cut replaces or accompanies the value cut.
- **No `W = 4` artifact is carried over** — not as a result, and not as the control arm of a new
  confirmatory claim (historical reproduction excepted).

**Execution order** (the draft's Appendix B, re-prioritized 2026-08-12): ① **B.7 protocol audit**
— completed *before* any run; the population/preprocessing definitions are the premise of every
experiment after it → ② **B.1 backbone gate** — `seq_v2` × 16 seeds + training-budget
equalization, in parallel with the **causal (past-only) GP baseline** → ③ **B.2 exploratory
model search** (attention-pooling family first) → ④ **B.3 `W = 2` minimal interpretable model**
(latent bottleneck) → ⑤ **B.4 size-scaling ceiling** → ⑥ **B.5 re-score and replace every
`W = 4`-based analysis**. **B.6** (kHz Mirnov feature table, incl. mode rotation frequency) is
asynchronous — it runs whenever the features arrive.

**Standing rules attached to the decision.**

1. **Test freeze.** All B.2 selection happens on val; TEST is scored once per candidate, in a
   confirmatory run whose decision rule is fixed beforehand.
2. **Claim-2 gate.** The causal GP may itself be the strongest deployable baseline (§8p showed the
   *offline* GP ties the model). Whether "beats every deployable causal method" survives is
   decided by B.1's causal-GP arm **before** any time is spent on B.2.
3. **Real-time claim adjudication.** "Instantly-reacting real-time" is only meaningful with
   microsecond-scale inference **and** kHz-stream inputs — the current fast-diagnostic inputs are
   decimated to 100 Hz, so new information arrives only every 10 ms regardless of inference speed.
   If either half is judged unattainable, the claim is removed and the contribution restated
   (pre-processor for profile-fitting pipelines; offline imputation + conformal uncertainty;
   baseline and protocol for multi-rate sensor-fusion work).

---

## 8w. B.7 protocol audit (2026-08-14) — the §8v gate is discharged

No training; the pre-experiment audit that §8v's order ① demands. The full 21-constraint
inventory is `ces_prediction/experiments/PROTOCOL_AUDIT.md`; the frozen protocol it feeds is
`ces_prediction/experiments/PREREGISTRATION_W2.md`; the re-verification script is
`ces_prediction/experiments/protocol_audit_stats.py`.

**Re-verification on the current 641 files** (all numbers in the audit's §1): the published
missingness ledger, MC lag-1 autocorrelation, and Te/NBI probe reproduce **exactly**; observed
`CES_TI` p99 = 2,089 eV reproduces, with > 3 keV = 1,197 rows = 0.53% of observed; the 0.5 s
segment threshold sits in the valley of a bimodal delta distribution (only 82 of ~247k inter-row
deltas fall in (0.1, 0.5) s, so the threshold is insensitive); `CES_VT` is recorded at 5 decimal
places with ≈ 4 × 10⁻⁵ minimum spacing between distinct values, so the held rule
(consecutive-equal ⇒ forward-fill) has no measurable false-positive channel — `CES_TI`'s
held = 1 / 226,991 is the empirical bound.

**Four (B) corrections, all pre-registered before any run:**
1. The fit-failure cut is inverted per §8v: implemented as a **load-time missing treatment**
   (`CES_TI_SPIKE_CUT_EV`, same four application points as the held rule — supervision targets,
   `ces_history`, normalization stats, interpolation anchors — so every arm is treated identically
   by construction). To be implemented and smoke-tested before B.1 trains (audit action #1).
2. Per-shot input standardization, adopted by §8s but never pinned in the common protocol, is now
   pinned: window family **OFF** (pairing consistency with the W = 2 controls), `seq_v2` **ON**
   (part of its definition), campaign analyses report the ON variant alongside.
3. The per-file sample cap 500, previously a window-sweep-only control, is promoted to a common
   frozen constant.
4. The B.1 gate control is replaced: §8t paired against the W = 4 held-free family; the gate pairs
   against the **W = 2** family instead, and both sides are retrained under the spike cut because
   the cut changes the population.

**Two (C) items stay scheduled:** the dataset-spec section is rewritten only when new data arrives
(re-running the audit's verification scripts first), and the routing rationale is refreshed with a
W = 2 held-free modality ablation in B.5 (the 2026-06-22 ablation was measured held-kept at W = 4;
§8f's history-0 and §8t's decomposition already re-ground it held-free).

**Verdict: gate discharged.** With this section and the two committed documents, §8v's order ①
is complete. Audit action #1 landed the same day (commit `3598760`): `CES_TI_SPIKE_CUT_EV` NaNs
spikes at load time before held detection, shared between the window dataset and the full-grid seq
loader, cache signature v5, recorded in `metrics.json`; 35/35 tests pass and a
W = 2 · held-free · cut = 3000 smoke train runs end to end. **B.1 may start.**

---

## 8x. B.1 backbone gate (2026-08-14) — seq_v2 wins all four pre-registered conditions; the causal GP restates claim 2

The first batch of the confirmed protocol (`W = 2` · held-free · 3 keV cut; runner
`experiments/b1_gate/run_b1_gate.py`, 2.2 h wall, artifacts `data/.b1_*`, verdict
`data/.b1_gate_summary.json`). Split pinned per seed to the frozen W = 2 sweep manifests
(recovered from the sweep runs' metrics.json); the test-isolation assert passed on all four seeds.

**Stage A — the `w2cut` window control family** (4 seeds, per-shot OFF). `skill_vs_pchip`
`CES_TI` +0.051 n.s. / **+0.203** / **+0.208** / **+0.231** → **3/4 PASS** (the cut population
strengthens interpolation: clean anchors); vs persistence **4/4 PASS** (mean +0.35); vs the
offline GP 0/4 (seed 42 significantly behind). `CES_VT` vs persistence **3/4 PASS**
(+0.39/+0.30/+0.35/+0.39, seed 123's CI includes 0) — under the confirmed protocol the `V_rot`
causal claim nearly resolves, a real upgrade from the held-kept era.

**Claim-2 gate (§8v rule 2).** Model vs **`gp_causal`**: `CES_TI` +0.080 PASS / −0.060 / +0.081
/ +0.064 → **1/4**, so per the pre-registered rule claim 2 is **restated** as "beats the standard
deployable baselines (persistence, AR)" and **"beat the causal GP" becomes a B.2 objective**. The
causal GP is decisively the strongest deployable baseline (e.g. seed 42 RMSE 164.3 vs persistence
197.2) — the honest deployable ladder now has a serious top rung.

**Stage B — `seq_v2` 16-run grid** (4 split × 4 init seeds, each paired against its split's
stage-A control). `CES_TI` paired skill: **16/16 positive, 13/16 individually significant**;
per-split init-means +0.129 / +0.059 / +0.078 / +0.058 (init spread ≪ split spread); pooled mean
**+0.081**, run-cluster bootstrap CI **[+0.067, +0.096]**. `CES_VT`: 8/16 significant wins
(splits 42 and 123 4/4 each), **0/16 significant deficits**.

**Stage C — budget equalization** (`CES_SEQ_FIXED_EPOCHS=1`, 10 epochs, final weights, no val
selection): `CES_TI` paired +0.063 / +0.033 / +0.045 / +0.030 — smaller than the early-stopped
arm but **4/4 positive**, so the advantage is attributable to the architecture, not the budget.

**Verdict (all four §4 conditions met): the backbone for B.3 is `seq_v2`.** §8t's cautious
"+0.045, 1/4 significant at 1/10 the cost" is now, under the confirmed protocol and against the
matching W = 2 control, "**+0.08 pooled, CI excludes zero, 13/16 significant, budget-equalized
sign 4/4**". B.2 exploration proceeds on the seq_v2 backbone with the added objective of beating
the causal GP; B.3's minimal interpretable model targets the seq_v2 structure.

---

## 8y. B.2 first candidate (2026-08-14) — v3 attention misses promotion; claim 2 is reinstated for the backbone

**Candidate.** `seq v3` = v2 + observation-masked causal attention readout (iter009's mechanism in
the sequence frame): each target path attends over the hidden states at its own past
fresh-observation steps (shifted mask, strict lower-triangular; zero-initialized output
projections so training starts at exactly the v2 function; routing preserved and structurally
re-verified). 396,930 params. Exploration was **val-only** (splits 42/7): paired `CES_TI` vs the
backbone +0.024*/+0.037* and vs `gp_causal` +0.170*/+0.132* — promoted to confirmation with the
decision rule **committed before any TEST scoring** (PREREGISTRATION_W2.md §6).

**Confirmation (TEST, 4 splits, init = split seed, vs the B.1 seq_v2 backbone):**

| seed | paired `CES_TI` (v3 − v2) | `CES_VT` deficit | v3 vs `gp_causal` `T_i` |
|---|---|---|---|
| 42 | +0.009 [−0.013, +0.031] | none | **+0.086 PASS** |
| 1 | +0.013 [−0.000, +0.026] | none | **+0.144 PASS** |
| 7 | **+0.033 [+0.004, +0.064] PASS** | none | **+0.166 PASS** |
| 123 | +0.020 [−0.011, +0.052] | none | **+0.123 PASS** |

**Verdict 1 — promotion FAILS by the pre-fixed rule** (needs ≥3/4 significant; got 4/4 positive,
1/4 significant). The val gains shrank on TEST — the exploration split pair was favorable. The
backbone stays **seq_v2**; the attention readout is a real but small mechanism (consistent
positive sign) and remains a B.4-adjacent scaling question, not the main model.

**Verdict 2 — claim 2 is reinstated, and for the ADOPTED model.** v3 clears the causal GP 4/4;
the direct cross-check (v2's `se_model` paired against `se_gp_causal` from the same scored rows,
population keys verified bit-identical) shows the **backbone itself beats the causal GP on
`CES_TI` 4/4**: +0.078* / +0.133* / +0.138* / +0.105* (`CES_VT` 4/4 positive, 2/4 PASS). Combined
with §8x, the deployable ladder now reads: **the window model ties the causal GP (1/4), the
sequence backbone beats it (4/4)** — the full-grid framing's history *reach* is what clears the
strongest deployable baseline, which is the reach narrative of §7.4/§7.6 landing exactly where it
was predicted to matter. Claim 2 for the seq_v2 backbone returns to "beats every deployable causal
method measured, including the causal GP".

---

## 8z. B.3 minimal interpretable model (2026-08-15) — an 8-number latent carries all of the backbone's `T_i` skill

**Question.** §8k priced the opacity of the *window* model at W = 4: a 1,258-parameter named-terms
anchor+Δ recovered ≈31% of its `T_i` margin, and the residual was read as "the recoverable
structure is non-linear, so an explainable model needs a different functional form, not more
linear terms". B.3 asks the same question of the *adopted* backbone (`seq_v2`, §8x) under the
confirmed protocol, with that lesson built in: interpretability comes from **structure**, not from
linearity of the encoder.

**Design** (`ces_prediction/experiments/b3_interp/`, model `experiments/seq/model_seq_b3.py`;
runners `run_b3_anchor.py` → `run_b3_explore.py` (val only) → `run_b3_confirm.py` (TEST once) →
`probe_b3.py`). Per target: `ŷ = carried_value + Σ_k w_k·z_k + b`, where the anchor is the data
layer's strictly-causal carry-forward (parameter-free persistence), `z ∈ [−1, 1]^K` is a tanh
latent from a small causal GRU (64 / 32 units), and the readout is **zero-initialised** so
training starts exactly at persistence and everything learned is, by construction, the
correction. The routing is seq_v2's, at the encoder: the `V_rot` GRU sees only the non-fast tail.
Consequences: (i) the decomposition into K + 1 named terms is *exact*, not an attribution
approximation; (ii) every number the readout can see is one of K probeable latents; (iii) the
correction is bounded by `Σ|w_k| + |b|` in target σ. `T_i` latent K ∈ {4, 8} was the one explored
variable (`V_rot` latent 4). **21,498 parameters** = 6% of seq_v2's 357,570.

Controls, all under W = 2 · held-free · 3 keV cut · cap 500: (a) the §8k anchor+Δ **retrained**
under the protocol (`.b3_anchor_s*`, val-only until the rule commit, TEST scored additively;
§8v forbids the W = 4 family as a control), (b) the B.1 backbone `.b1_seqv2_s{seed}_i{seed}`,
(c) the w2cut window family (reported). Row-for-row pairing verified by
`paired_model_compare.py`'s bit-identical `se_pchip` guard on every pair.

**Exploration (val, splits 42/7; test frozen).** K = 4: paired `T_i` vs the backbone
−0.083\* / −0.022 ns; K = 8: −0.035\* / +0.016 ns → **b3k8** by the pre-stated rule. Both hit the
seq family's 30-epoch cap (`best_epoch` 29–30/30) while the backbone had been terminated by the
patience rule at 14–19 epochs; raising the cap to 60 let the same rule fire at 56 on split 42 and
moved the paired number from −0.035\* to −0.005 ns. The candidate was therefore defined with a
**non-binding 100-epoch cap** so both arms stop by the identical rule (patience 6 on val masked
MSE); the confirmatory runs terminated at 56 / 57 / 88 / 54. The decision rule was committed to
`PREREGISTRATION_W2.md` §6 (commit `df79522`) before any TEST scoring.

**Confirmation (TEST, 4 splits, init = split seed):**

| seed | `T_i` vs PCHIP | vs anchor+Δ (retrained) | vs seq_v2 backbone | vs w2cut window | vs causal GP | epochs |
|---|---|---|---|---|---|---|
| 42 | +0.166 PASS | **+0.351\*** | −0.009 [−0.037, +0.017] | +0.122\* | +0.069 PASS | 56 |
| 1 | +0.245 PASS | **+0.401\*** | −0.005 [−0.033, +0.025] | +0.053\* | +0.129 PASS | 57 |
| 7 | +0.276 PASS | **+0.418\*** | +0.026 [−0.013, +0.065] | +0.087\* | +0.161 PASS | 88 |
| 123 | +0.262 PASS | **+0.411\*** | −0.004 [−0.058, +0.045] | +0.040 ns | +0.101 PASS | 54 |

**Verdict — both pre-registered branches met.** (1) *Interpretable rung*: `T_i` significant win
over the anchor 4/4, `V_rot` significant deficit vs the anchor 0/4 (in fact `V_rot` wins 4/4).
(2) *Backbone tolerance*: mean paired `T_i` = **+0.002** against a −0.05 tolerance — every CI
contains 0, one split is positive. Reported, not gated: `T_i` vs PCHIP PR4 **PASS 4/4**, vs
persistence PASS 4/4, vs the offline GP tie 4/4 (as for every model), vs the window family
positive 4/4 (3/4 significant), vs the causal GP **PASS 4/4** — the 21k model inherits the
backbone's claim 2. `V_rot` vs persistence PASS 4/4.

**The complexity ladder, re-priced under the confirmed protocol** (TEST `T_i` skill vs PCHIP,
4-seed mean; fraction of the persistence → backbone gap recovered):

| rung | params | mean skill | recovered |
|---|---:|---:|---:|
| persistence | 0 | −0.264 | 0% |
| anchor+Δ (§8k form, retrained at W = 2) | 1,258 | −0.261 | **1%** |
| **b3k8 (this section)** | **21,498** | **+0.237** | **≈100%** (98 / 99 / 104 / 99) |
| w2cut window model | 201,258 | +0.173 | 87% |
| seq_v2 backbone | 357,570 | +0.236 | 100% |

> **Restated 2026-08-16 (§8ab):** this ladder is a **cut-population** statement. Re-run in the
> spike-*inclusive* population (§8ab block 3), b3k8 still beats the retrained anchor 4/4 with no
> `V_rot` deficit, but loses to the backbone −0.160\* / −0.200\* / −0.203\* / −0.214\* (mean −0.194
> against the −0.05 tolerance) and to the window family on 3/4 splits: the ≈ 1% spike-anchor rows
> carry 73–83% of its `T_i` squared error there, and a bounded correction cannot recover a spiked
> anchor — the same mechanism this section documents for `V_rot` below. "21k = backbone" is
> therefore conditional on the 3 keV cut.

*(All rungs computed from the frozen TEST npz files. Two backbone dirs' unsuffixed
`comparison_metrics.json` had been overwritten by a later val re-score — caught 2026-08-15,
reports regenerated with population verified bit-identical, and `eval_seq.py` /
`compare_baselines.py` now write split-tagged reports and never overwrite a TEST report with
val; every paired verdict is npz-based and unaffected.)*

Two things this table says that §8k could not. First, **the §8k anchor was a W = 4 artefact**: at
W = 2 its slope term needs two observed history rows and never fires, and window statistics over
one or two rows are noise, so it collapses onto persistence — condition (1) above is therefore
effectively "beats persistence-level", the weak condition; the informative comparisons are (2),
the causal GP and the window family. Second, **the backbone's entire `T_i` skill survives
compression to eight bounded numbers plus persistence** — the opacity that §8k priced at 69% of
the margin was the price of the *window* model's form, not of the task. What the task needs is
the full-grid causal state (§8t/§8x), and that state is small.

**Pre-registered measurements (`probe_b3.py`, `data/.b3_probe_summary_b3k8.json`).**
① *Linear probes* (OLS fit on ~155k train-file grid steps, R² on ~35–39k TEST-file steps —
inputs only, no targets touched): the `T_i` latent linearly encodes the **last observed `T_i`**
(R² 0.54 / 0.75 / 0.48 / 0.56) and the **ECEI `T_e` proxy** (0.48 / 0.31 / 0.40 / 0.40); local
BES activity is barely decodable (0.09–0.13) and staleness hardly at all (0.01–0.07). No single
latent dimension carries any quantity (best per-dim R² 0.14–0.37): the representation is
distributed across the eight. The `V_rot` latent encodes carried `V_rot` (0.46–0.83) and staleness
(0.05–0.63). ② *Exact decomposition* on TEST steps: the learned correction has σ ≈ 0.46–0.57
against an anchor σ ≈ 1.0–1.1 and accounts for 25–39% of `T_i` prediction variance; readout
weights |w_k| ≈ 0.4–0.65, so the correction is bounded at 4.2–5.2σ (`V_rot`: 3.0–4.7σ, 9–33% of
variance). ③ *Structural routing*: perturbing the 15 fast channels leaves the `V_rot` output
**bit-identical** and changes `T_i` on all four checkpoints.

**`V_rot` — the bounded correction meets fit-failure spikes.** Against the backbone, `V_rot`
paired skill is −0.55\* / −0.14\* / −0.38 ns / −0.51\* — but rows whose *persistence* error exceeds
1,000 (i.e. whose previous `V_rot` observation is a spike of thousands) number 1 / 0 / 3 / 4 per
split (≤ 0.03% of rows) and carry 28 / 0 / 64 / 72% of b3's `V_rot` squared error; excluding them
the paired numbers are −0.12 / −0.14 / 0.00 / 0.00. Mechanism: `anchor + (≤ 4σ) correction` cannot
recover from a spiked anchor, and persistence, anchor+Δ and b3 fail those rows identically while
the backbone's unbounded head partly recovers (on the exploration split 7 **val** set the
backbone's `V_rot` skill vs PCHIP is +0.876 for exactly this reason — one recovered spike-anchor
row, not physics). `T_i` is immune because the pre-registered 3 keV cut removes such anchors
before they are carried. This is §8q's lesson recurring on the other target: **`V_rot` has
fit-failure spikes too, and any persistence-anchored method's MSE is hostage to them.** No
protocol change is made here; a `V_rot` spike audit and cut/sensitivity rule are recorded for
승상님's decision (B.7 follow-up), and the same rows would need the same treatment in every arm.

**What this section does not show.** It does not make b3k8 the main model — the backbone stays
seq_v2 by §8x; b3k8 is the interpretable rung *and* an ablation stating what the backbone's `T_i`
skill consists of. It does not claim `V_rot` parity. And the probe R² are linear-decodability
numbers, not causal attributions.

Artifacts: `data/.b3_anchor_s*` (val + test), `data/.b3_b3k{4,8}_s{42,7}[_e60]` (val
exploration), `data/.b3_explore_summary{,_e60}.json`, `data/.b3c_b3k8_s*` (TEST + probes),
`data/.b3c_b3k8_summary.json`, `data/.b3_probe_summary_b3k8.json`.

---

## 8aa. B.4 size-scaling ceiling (2026-08-15) — the `T_i` curve is flat across a 26× parameter range

**Question.** With the backbone fixed (§8x) and its `T_i` skill shown to survive compression to
eight latents (§8z), does *more* capacity buy anything on this input set — or is the input set,
not the model, the ceiling? Pre-registered reading (`PREREGISTRATION_W2.md` §6 B.4, committed
before the run): one controlled variable, the seq_v2 `T_i`-encoder width `hidden_ti` ∈ {24, 40,
80, 160, 260} (34k / 49k / 114k / 358k / 879k params; 260 is the last width under the 1 M cap);
`V_rot` branch, heads, depth, data treatment and training constants fixed; the 160 point *is* the
B.1 backbone run; new widths train with a non-binding 100-epoch cap so every point stops by the
same patience rule (terminations 14–49 epochs). TEST scored once per point, paired against the 160
point and the w2cut window family; no backbone re-selection.
Runner `experiments/b4_scale/run_b4_scale.py`, 82 min wall, `data/.b4_w{24,40,80,260}_s*`,
verdict `data/.b4_scale_summary.json`.

**Curve (TEST `T_i` skill vs PCHIP; paired vs the 160-unit backbone, \* = significant):**

| `hidden_ti` | params | s42 | s1 | s7 | s123 | mean | paired vs 160 (s42 / s1 / s7 / s123) | mean Δ | PR4 vs PCHIP | vs causal GP |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---|---|
| 24 | 34,162 | +0.156 | +0.242 | +0.251 | +0.271 | +0.230 | −0.021\* / −0.008 / −0.008 / +0.009 | −0.007 | 4/4 | 4/4 |
| 40 | 49,170 | +0.161 | +0.245 | +0.258 | +0.278 | +0.236 | −0.015 / −0.005 / +0.002 / +0.019 | +0.000 | 4/4 | 4/4 |
| 80 | 113,570 | +0.143 | +0.248 | +0.279 | +0.268 | +0.235 | −0.037\* / −0.001 / +0.030\* / +0.005 | −0.001 | 4/4 | 3/4 |
| **160** | 357,570 | +0.174 | +0.248 | +0.257 | +0.264 | +0.236 | (reference) | — | 4/4 | 4/4 |
| 260 | 878,570 | +0.165 | +0.243 | +0.281 | +0.232 | +0.230 | −0.011 / −0.007 / +0.033\* / −0.044 | −0.008 | 4/4 | 3/4 |

`V_rot` (branch fixed at 64 units) does not move with `T_i` width: mean skill +0.250…+0.254 at
every width, paired vs 160 within ±0.02 and non-significant on 19/20 pairs — the internal
consistency check the rule asked for.

**Verdict (descriptive, as pre-registered).** ① **Ceiling reached at or below the backbone
width**: 260 beats 160 significantly on 1/4 splits (< 3/4) and its mean Δ is −0.008. ② **No
knee down to the smallest width measured**: no width < 160 loses significantly on ≥ 3/4 splits
(the two significant losses are single splits: 24 on s42, 80 on s42); the mean curve is
+0.230 → +0.236 → +0.235 → +0.236 → +0.230 across a **26× parameter range**. ③ Every width passes
PR4 vs PCHIP 4/4 and beats the causal GP on 3–4/4.

**Reading.** This is the "flat curve = performance ceiling of this input set" outcome the
pre-registration named. Together with §8z (21k latent model = backbone) it says the `T_i` skill
available from {BES, ECEI, MC at 100 Hz + CES history + time} is exhausted by ~50k parameters of
causal recurrent state; the remaining variance across splits (s42 +0.14…+0.17 vs s123 +0.23…+0.28)
is split variance, not capacity. Two consequences for the thesis: the model-size axis is closed
(B.4 does not motivate a larger model), and the levers that remain are *inputs* — NBI for `V_rot`
(§9.4), kHz Mirnov features (B.6), CES fit-quality metadata (§8v) — not architecture.

**What it does not show.** Depth was not varied (one variable per batch); the width ladder is
capped at 879k by the project's 1 M rule, so "no gain up to 879k" is the measured statement, not
"no gain ever"; and w80/w260's isolated significant wins on s7 (+0.030\*/+0.033\*) are single-split
effects with the opposite sign on s42, exactly the split-variance pattern §8x recorded.

---

## 8ab. B.5 full re-score (2026-08-16) — every W = 4-based analysis replaced, in both populations

**Question.** §8v demanded that every quantitative analysis of the draft — headline, MNAR
reweighting, campaign (temporal) split, conformal intervals, peak / large-gap strata, the
interpretability ladder, the modality ablation behind the `V_rot` routing — be recomputed once
under the confirmed protocol (`W = 2` · held-free · cap 500), and §1.1 of the pre-registration
demands it in **two populations**: the **cut** regime (`CES_TI_SPIKE_CUT_EV = 3000`, "p99") and
the **inclusive** regime (cut 0, "p100"), each internally consistent across training targets,
history inputs, interpolation anchors and evaluation, and identical for every arm inside it. An
unqualified claim must hold in both; anything that holds in one is reported conditional on that
population.

**Design** (`experiments/b5_rescore/run_b5.py`, plan committed 2026-08-15 before the run,
`PREREGISTRATION_W2.md` §6 B.5; run in two sessions — interrupted 2026-08-16 00:40 at
`cut_sens` 4/8 and resumed from the per-stage artifacts, no stage re-run; verdict
`data/.b5_summary.json`, all nine analysis blocks `ok`). Models: the adopted backbone `seq_v2`
(§8x), the `W = 2` window family as the paired control, and the interpretable rung `b3k8` +
retrained anchor+Δ (§8z). No new decision rule — every verdict below uses the rule already fixed
for that analysis. TEST scored once per new run.

| population | backbone `seq_v2` | window control (`W = 2`) | split | ladder |
|---|---|---|---|---|
| **cut** (3 keV) | B.1 `.b1_seqv2_s{s}_i{s}` (val scored additively for conformal) | B.1 `.b1_w2cut_s*` | `.b1_w2cut_split_s*` | §8z `.b3c_b3k8_s*`, `.b3_anchor_s*` |
| **inclusive** (cut 0) | **new** `.b5i_seqv2_s*` (4 seeds, val + TEST; rule-terminated 14–22 epochs) | frozen no-cut W = 2 sweep `.wsweep_hf_w2_s*` re-scored with the current harness → `.b5i_w2_s*` (10 population keys bit-identical vs the frozen npz on 4/4 seeds, `se_model` drift 0.000) | `.b1_manifest_s*` | **new** `.b5i_b3k8_s*` (cap 100, terminated 55–59), `.b5i_anchor_s*` |

Plus: campaign split (65/20/15 by shot number — train 30801–31991 / val 32002–32310 / test
32312–32751, 416 / 128 / 97 files, `.campaign_split_manifest.json` rebuilt) with window OFF /
window ON (cut only) / `seq_v2` × 4 init seeds in each population (20 runs,
`.b5_camp_{cut,incl}_{win,winps,seq}_s*`); cut-threshold sensitivity `.b5c{2500,4000}_seqv2_s*`
(8 runs, self-population scoring); modality ablation of the window family at evaluation
(`CES_ABLATE ∈ {no_fast, no_history}`, zeroed inputs, no retraining, copies
`.b5_abl_{pop}_{abl}_s*`, 16 scorings). CPU analyses (headline, coverage / PR2, ladder,
conformal, peak + large-gap, campaign, cut sensitivity, ablation, MNAR) all live in
`.b5_summary.json`; every number below is read from it.

### 1. Headline (TEST, 4 splits, PR4 = shot-clustered paired bootstrap; \* = CI excludes 0)

`T_i` skill vs PCHIP per seed (42 / 1 / 7 / 123), PR4 pass counts vs PCHIP · persistence ·
causal GP, and the row-paired `seq_v2 − window` skill:

| population | arm | s42 | s1 | s7 | s123 | mean | vs PCHIP | vs persistence | vs causal GP | `seq_v2 − window` |
|---|---|---:|---:|---:|---:|---:|---|---|---|---|
| cut | **`seq_v2`** | +0.174\* | +0.248\* | +0.257\* | +0.264\* | **+0.236** | **4/4** | 4/4 | **4/4** | +0.130\* / +0.058\* / +0.062 / +0.044 |
| cut | window | +0.051 | +0.203\* | +0.208\* | +0.231\* | +0.173 | 3/4 | 4/4 | 1/4 | — |
| inclusive | **`seq_v2`** | +0.225\* | +0.238\* | +0.292\* | +0.316\* | **+0.268** | **4/4** | 4/4 | **4/4** | +0.053\* / +0.024 / +0.047\* / +0.029 |
| inclusive | window | +0.181\* | +0.219\* | +0.258\* | +0.296\* | +0.238 | 4/4 | 4/4 | 3/4 | — |

`V_rot`: `seq_v2` vs PCHIP +0.390\* / +0.183 / +0.135 / +0.305 (cut, **1/4**) and +0.384\* /
+0.195\* / +0.132 / +0.304 (inclusive, **2/4**); vs persistence 3/4 in both populations
(+0.30…+0.50); vs the causal GP 2/4 in both. Window `V_rot` vs PCHIP 0/4 in both.

**Unconditional (holds in both populations):** the backbone's `T_i` beats PCHIP **4/4 + 4/4**,
beats the causal GP **4/4 + 4/4** (claim 2 stands, §8y), beats persistence 4/4 + 4/4; and it is
positive against the window family on 8/8 pairs (2/4 significant in each population — the
confirmatory evidence for the backbone remains §8x's 16-run grid; this 4-run slice is
consistent with it, cut mean +0.074 vs pooled +0.081). **Conditional:** the *window* model's
`T_i` PR4 vs PCHIP is 3/4 in the cut regime and 4/4 inclusive. **Not established in either:**
`V_rot` vs offline interpolation (1/4, 2/4); `V_rot` vs persistence stays 3/4 in both — the
`V_rot` verdict remains "tie with interpolation, ahead of persistence on most splits".

Two things the pair of populations says that one alone would not. The inclusive numbers are
*higher* than the cut numbers for every arm (backbone +0.268 vs +0.236, window +0.238 vs
+0.173, and the `seq_v2 − window` gap shrinks from +0.074 to +0.038): the spikes poison the
interpolation anchors, so the offline baseline is weaker in p100 — exactly the §1.1 rationale
for not using p100 alone. And the cut regime is the one where the *fast-diagnostic* content of
the model is isolated (ablation, block 8 below).

### 2. Coverage and PR2 fallback (TEST, per split 42 / 1 / 7 / 123; identical for both arms)

| population | target | scored rows | Δt > 15 ms rows | Δt > 45 ms rows | PR2 persistence-fallback rate |
|---|---|---|---|---|---|
| cut | `T_i` | 32,589 / 35,582 / 35,759 / 34,038 | 603 / 974 / 815 / 1,030 | 120 / 124 / 97 / 119 | 0.4 / 0.3 / 0.3 / 0.3 % |
| cut | `V_rot` | 10,463 / 13,247 / 14,477 / 13,968 | 91 / 135 / 130 / 110 | 3 / 2 / 6 / 3 | **43 / 44 / 40 / 42 %** |
| inclusive | `T_i` | 32,721 / 35,743 / 35,916 / 34,271 | 584 / 959 / 803 / 988 | 115 / 117 / 90 / 107 | 0.4 / 0.3 / 0.3 / 0.3 % |
| inclusive | `V_rot` | 10,461 / 13,243 / 14,475 / 13,966 | 89 / 131 / 128 / 108 | 3 / 2 / 6 / 3 | 43 / 44 / 40 / 42 % |

The cut removes 132–233 scored `T_i` rows per split (0.4–0.7%). PR3 holds everywhere. The
`V_rot` fallback rate is the number the pre-registration asked to be surfaced: for 40–44% of
scored `V_rot` rows the interpolation baseline has no future neighbour and degrades to
persistence, so "vs PCHIP" for `V_rot` is 40% "vs persistence" — one more reason the `V_rot`
comparison to offline interpolation carries little information at W = 2 (block 9 gives the
MNAR side of the same fact).

> **Correction (2026-10-13, §8bb P02).** The 40–44 % figure is a counting artifact: `eval_seq.py`
> increments `n_total` for *every* sample and both targets before the per-target keep mask is
> applied, so `future_neighbor_fraction` for `V_rot` is dominated by rows where `V_rot` is not
> scored at all. Measured on the frozen npz over scored rows (all three two-sided arms equal to
> persistence), the PR2 fallback share is **0.3–0.5 % of scored `V_rot` rows** (0.47 / 0.35 /
> 0.35 / 0.34 %) and 0.2 % of `T_i` rows, on all four splits. The sentence "vs PCHIP for `V_rot`
> is 40 % vs persistence" does **not** hold for the scored population and should not be quoted;
> the `V_rot` tie with interpolation is explained by P02 (unobserved torque ⇒ history-only
> optimum, bridge variance in the bulk, tail-only aggregate gain), not by fallback.

### 3. Interpretability ladder in both populations — §8z is a cut-population statement

TEST `T_i` skill vs PCHIP (4-seed mean) and the paired verdicts of the §8z rule:

| population | persistence | anchor+Δ | **b3k8** | window | `seq_v2` | b3 − anchor (`T_i`) | b3 − `seq_v2` (`T_i`) | `V_rot` deficit vs anchor | rung / tolerance |
|---|---:|---:|---:|---:|---:|---|---|---|---|
| cut (§8z) | −0.264 | −0.261 | **+0.237** | +0.173 | +0.236 | +0.35\* / +0.40\* / +0.42\* / +0.41\* (4/4) | −0.009 / −0.005 / +0.026 / −0.004, mean **+0.002**, all CIs ∋ 0 | 0/4 (b3 wins 4/4) | **met / met** |
| inclusive | −0.288 | −0.287 | **+0.126** | +0.238 | +0.268 | +0.33\* / +0.32\* / +0.29\* / +0.34\* (4/4) | **−0.160\* / −0.200\* / −0.203\* / −0.214\***, mean **−0.194**, all CIs exclude 0 | 0/4 (b3 wins 4/4) | **met / FAILED** |

The rung condition (beats the retrained anchor 4/4 with no `V_rot` deficit) holds in both
populations; the backbone-tolerance condition (mean ≥ −0.05) holds **only in the cut
population**. In p100 the 21k model sits below the window family too (paired −0.098 / −0.171\* /
−0.147\* / −0.178\*, 3/4 significant). **Mechanism, measured from the frozen npz:** rows whose
persistence error exceeds 2 keV — i.e. whose carried anchor is a fit-failure spike — are 0.64 /
0.86 / 0.78 / 1.28% of the inclusive TEST rows and carry **73 / 73 / 83 / 82%** of b3's `T_i`
squared error; the same rows carry 76–83% of the backbone's, 70–82% of the window model's and
73–79% of PCHIP's. A bounded correction (`anchor + Σ|w_k| ≤ 4–5σ`) cannot recover a spiked
anchor; the unbounded heads of `seq_v2` and the window model partly can. So the §8z verdict "the
backbone's entire `T_i` skill survives compression to eight numbers" is **conditional on the
cut population** — in p100 the interpretable rung costs −0.16…−0.21 against the backbone, and
that price is the price of *not* having the cut, not of the latent size (it is the same
mechanism §8z documented for `V_rot`). §8z's ladder table has been annotated accordingly.

### 4. Conformal intervals (backbone `seq_v2`, α = 0.10, calibrated on the run's own val, TEST)

4-seed means of the Winkler score (lower is better) with the coverage range across seeds;
identical procedure for the two baselines:

| population | target | mode | model (coverage) | PCHIP (coverage) | persistence (coverage) | model best, seeds |
|---|---|---|---:|---:|---:|---|
| cut | `T_i` | global | **1,272** (0.87–0.92) | 1,554 (0.85–0.91) | 1,727 (0.86–0.91) | 4/4 vs each |
| cut | `T_i` | mondrian | **1,224** | 1,490 | 1,647 | 4/4 |
| inclusive | `T_i` | global | **2,290** (0.87–0.92) | 2,851 (0.85–0.91) | 3,120 (0.86–0.91) | 4/4 |
| inclusive | `T_i` | mondrian | **2,186** | 2,720 | 2,978 | 4/4 |
| cut | `V_rot` | global | **150** (0.91–0.94) | 164 (0.92–0.94) | 179 (0.92–0.94) | 4/4 |
| inclusive | `V_rot` | global | **149** (0.92–0.94) | 163 (0.92–0.94) | 179 (0.92–0.94) | 4/4 |

Coverage is at or just under the 0.90 nominal for `T_i` (0.87–0.92; seed 1 is the low split for
every arm) and slightly over for `V_rot`; the model's Winkler score is the best of the three arms
in **every** population × target × mode × seed cell (32/32). In the inclusive regime the
model's `T_i` intervals are actually *wider* than PCHIP's (half-width 224–255 vs 211–241 eV,
global) and still score better — the miss penalty is what the spikes inflate, and the model
misses less. Mondrian (per-Δt-bin) calibration tightens every `T_i` arm by ≈ 4–5%, leaves `V_rot`
unchanged, and changes no verdict. Unconditional.

### 5. Peak strata and large-gap regime

Peak-stratified PR4 (per-seed skill vs PCHIP, `is_peak` key, ≥ 4,000 peak rows per split for
`T_i`, ≥ 2,400 for `V_rot`):

| stratum | arm | cut: skill range · PASS | inclusive: skill range · PASS |
|---|---|---|---|
| `T_i` peak | `seq_v2` | +0.45…+0.61 · **4/4** | +0.62…+0.72 · **4/4** |
| `T_i` peak | window | +0.36…+0.49 · 4/4 | +0.57…+0.65 · 4/4 |
| `T_i` non-peak | `seq_v2` | +0.09…+0.20 · **4/4** | +0.06…+0.19 · 2/4 |
| `T_i` non-peak | window | −0.05…+0.17 · 3/4 | +0.06…+0.18 · 2/4 |
| `V_rot` peak | `seq_v2` | +0.54…+0.79 · 2/4 | +0.54…+0.79 · 2/4 |
| `V_rot` non-peak | `seq_v2` | −0.07…+0.15 · 0/4 | −0.06…+0.15 · 0/4 |

The transient-capture claim is **unconditional** for `T_i` (peak rows PASS 4/4 in both
populations, for both models, at +0.45…+0.72 against an interpolator that has the future
neighbour); the non-peak `T_i` margin is conditional (backbone 4/4 cut, 2/4 inclusive). `V_rot`
skill lives entirely in the peak stratum (+0.54…+0.79 point estimates on every split, 2/4
significant) and is nil elsewhere — the same "three regimes" picture as §8r, now under the
confirmed protocol.

Large-gap, 4-seed pooled by physical shot (skill vs PCHIP [CI]):

| stratum | arm | cut | inclusive |
|---|---|---|---|
| `T_i` Δt ≤ 15 ms (n ≈ 135k, 301 shots) | `seq_v2` | +0.239 [+0.197, +0.274]\* | +0.299 [+0.244, +0.347]\* |
| `T_i` Δt > 15 ms (n = 3,422 / 3,334, 265 / 263 shots) | `seq_v2` | +0.268 [+0.187, +0.337]\* | +0.206 [+0.108, +0.290]\* |
| `T_i` Δt > 45 ms (n = 460 / 429, 104 / 101 shots) | `seq_v2` | +0.267 [+0.092, +0.414]\* | −0.004 [−0.304, +0.246] |
| `T_i` Δt > 45 ms | window | +0.325 [+0.160, +0.471]\* | +0.062 [−0.231, +0.317] |
| `V_rot` Δt > 15 ms (n = 466 / 456, 130 shots) | `seq_v2` | +0.418 [+0.104, +0.680]\* | +0.432 [+0.128, +0.696]\* |
| `V_rot` Δt > 45 ms (n = 14) | — | not scored (< 50 rows) | not scored |

The `T_i` gain persists into the > 15 ms regime in both populations (unconditional) and into
the > 45 ms regime only in the cut population (429 rows / 101 shots in p100 is where a handful of
spike rows dominate a small stratum). `V_rot` beats interpolation in the > 15 ms stratum in both
populations — the one `V_rot` regime with a positive unconditional verdict, on 130 shots.

### 6. Campaign (temporal) split — the offline-superiority claim survives for the backbone

`T_i` skill vs PCHIP per init seed (42 / 1 / 7 / 123) on the temporal TEST block (97 files,
shots 32312–32751); \* = PR4 PASS:

| population | arm | s42 | s1 | s7 | s123 | vs PCHIP | vs persistence | vs causal GP | paired vs window OFF |
|---|---|---:|---:|---:|---:|---|---|---|---|
| cut | window OFF | +0.027 | +0.091\* | −0.001 | +0.061\* | 2/4 | 4/4 | 0/4 (−0.05 / +0.02 / −0.08 / −0.01) | — |
| cut | window ON (per-shot) | +0.103\* | +0.107\* | +0.094\* | +0.107\* | **4/4** | 4/4 | — | +0.078\* / +0.018 / +0.095\* / +0.049\* |
| cut | **`seq_v2`** | +0.187\* | +0.174\* | +0.181\* | +0.177\* | **4/4** | 4/4 | **4/4** (+0.12 / +0.11 / +0.12 / +0.11) | +0.164\* / +0.091\* / +0.182\* / +0.124\* |
| inclusive | window OFF | +0.014 | +0.047 | +0.055 | +0.089 | **0/4** | 4/4 | 0/4 | — |
| inclusive | **`seq_v2`** | +0.173\* | +0.202\* | +0.198\* | +0.184\* | **4/4** | 4/4 | **4/4** (+0.13 / +0.16 / +0.16 / +0.14) | +0.161\* / +0.163\* / +0.151\* / +0.104\* |

`V_rot` on the temporal split: `seq_v2` vs persistence **4/4** in both populations (+0.33 cut,
+0.31…+0.33 inclusive), vs PCHIP 0/4 (+0.19…+0.23 n.s.); window OFF vs persistence 0/4;
`seq_v2 − window` `V_rot` +0.128\* / +0.131\* / +0.175\* / +0.136\* (cut) and +0.144\* /
+0.101\* / +0.105\* / +0.063\* (inclusive) — 8/8 significant.

This re-adjudicates §8n/§8o. The **window** model's advantage over offline interpolation does
collapse under campaign shift (2/4 cut, 0/4 inclusive — the §8n finding reproduces at W = 2 in
both populations), and per-shot standardization repairs it in the cut regime exactly as §8s
found (ON 4/4, ON − OFF positive 4/4, 3/4 significant). But the **backbone** does not collapse:
`seq_v2` beats PCHIP 4/4, beats the causal GP 4/4 and beats the window family 4/4 in **both**
populations on a test block of shots that all post-date every training shot. The offline
superiority claim, which §8n had to restate as "survives only against causal baselines", is
restored for the adopted model — unconditionally. Caveat that stays: the four campaign passes
are init seeds on one temporal test block (97 files), so this is one temporal split's evidence,
not four splits'; and 2/4 of the cut `seq_v2` campaign runs stopped at the 30-epoch cap.

### 7. Cut-threshold sensitivity (backbone, cut population, self-population scoring)

| threshold | `T_i` skill vs PCHIP (42 / 1 / 7 / 123) | mean | PR4 vs PCHIP | vs causal GP | `V_rot` vs PCHIP mean · PASS |
|---|---|---:|---|---|---|
| 2.5 keV | +0.160\* / +0.254\* / +0.249\* / +0.255\* | +0.230 | 4/4 | 4/4 | +0.252 · 1/4 |
| **3 keV** (protocol) | +0.174\* / +0.248\* / +0.257\* / +0.264\* | +0.236 | 4/4 | 4/4 | +0.253 · 1/4 |
| 4 keV | +0.176\* / +0.234\* / +0.258\* / +0.260\* | +0.232 | 4/4 | 4/4 | +0.257 · 2/4 |

The threshold is immaterial across 2.5–4 keV: means within ±0.006, every PR4 verdict identical.
(7/8 of the new threshold runs reached the 30-epoch cap with `best_epoch` 25–29, while the 3 keV
reference stopped by rule at 14–25 — the neighbouring thresholds are if anything lower bounds.)

### 8. Modality ablation of the window family (evaluation-time, no retraining; audit C-2)

Paired skill vs the un-ablated run (42 / 1 / 7 / 123; \* = significantly worse):

| ablation | target | cut | inclusive |
|---|---|---|---|
| `no_fast` | `T_i` | −0.246\* / −0.375\* / −0.418\* / −0.425\* (skill → −0.10…−0.18) | −0.034\* / −0.043 / −0.033 / −0.089\* (skill → +0.15…+0.23) |
| `no_fast` | `V_rot` | **+0.000 / +0.000 / +0.000 / +0.000** | **+0.000 / +0.000 / +0.000 / +0.000** |
| `no_history` | `T_i` | −4.6\* / −1.8\* / −2.3\* / −1.9\* | −1.5\* / −3.4\* / −1.2\* / −1.1\* |
| `no_history` | `V_rot` | −5.4\* / −6.3\* / −1.8\* / −2.3\* | −6.9\* / −7.8\* / −1.9\* / −2.2\* |

Two routing facts are re-grounded held-free at W = 2 in both populations: zeroing the fast
channels leaves the `V_rot` output **bit-identical** (the routing is structural, not learned),
and zeroing the history collapses both targets (skill −1…−8: neither target is predictable from
100 Hz diagnostics without the CES anchor). The third row is the informative one for the
population question. In the cut regime the window model's `T_i` margin over PCHIP is *made of*
fast-diagnostic information: without BES/ECEI/MC it falls below the interpolator (−0.10…−0.18).
In the inclusive regime a history-only model still beats PCHIP by +0.15…+0.23 and the fast
channels add only 0.03–0.09 (2/4 significant) — because PCHIP's anchors are spiked and a learned
model discounts them where an interpolator cannot. **The p100 margin therefore contains a
spike-robustness component that is not fast-diagnostic information; the cut population is the one
that isolates what the fast diagnostics contribute.** That is the measured argument for keeping
the two-population report rather than a p100-only headline.

### 9. MNAR reweighting (W = 2 in-domain: nearest past observation within 2 rows)

Strata = Δt bin (15 / 25 / 45 ms) × input-only activity flag; weights from the genuinely-missing
rows of the same split (population-consistent: the cut is applied to the weight grid too);
coverage = missing-mass of the strata that survive the ≥ 30-scored-rows floor.

| population | target | in-domain fraction of missing rows | coverage | `seq_v2` reweighted vs PCHIP (42 / 1 / 7 / 123) | PASS | vs persistence PASS | window vs PCHIP PASS |
|---|---|---|---|---|---|---|---|
| cut | `T_i` | 0.54 / 0.61 / 0.63 / 0.67 | 0.99–1.00 | +0.140 / +0.164\* / +0.203 / +0.283\* | 2/4 | 4/4 | 2/4 |
| inclusive | `T_i` | 0.54 / 0.62 / 0.64 / 0.68 | 0.99 | +0.140\* / +0.217\* / +0.167\* / +0.221\* | **4/4** | 4/4 | 4/4 |
| cut | `V_rot` | **0.048 / 0.064 / 0.049 / 0.044** | 0.73–0.76 | +0.115 / +0.126 / +0.177 / +0.280 | 0/4 | 1/4 | 0/4 |
| inclusive | `V_rot` | 0.048 / 0.064 / 0.049 / 0.044 | 0.73–0.76 | +0.118 / +0.141 / +0.179 / +0.321\* | 1/4 | 2/4 | 0/4 |

Reweighted to the genuinely-missing distribution, the backbone's `T_i` margin over PCHIP is
significant on 2/4 splits in the cut regime and 4/4 inclusive (point estimates +0.14…+0.28 in
both; the reweighting bootstrap is wide because the > 45 ms strata are thin), and over
persistence 4/4 everywhere — the §8i pattern (unconditional vs causal, conditional vs offline)
holds at W = 2. The coverage numbers the pre-registration asked for: only **54–68% of
genuinely-missing `T_i` rows** are in the W = 2 model's domain at all (a past observation within
2 rows), and for `V_rot` **4–6%** — `V_rot` gaps are long blocks, so the MNAR-reweighted `V_rot`
estimate answers a question about 5% of the missing mass and is not informative at W = 2.

### Verdict — what is unconditional under the confirmed protocol

| claim | cut | inclusive | status |
|---|---|---|---|
| backbone `T_i` beats offline PCHIP (PR4) | 4/4 | 4/4 | **unconditional** |
| backbone `T_i` beats the causal GP (claim 2) | 4/4 | 4/4 | **unconditional** |
| backbone `T_i` beats persistence / window family (sign) | 4/4 / 4/4 | 4/4 / 4/4 | **unconditional** (window gap significant 2/4 each; §8x carries the 16-run proof) |
| `T_i` transient (peak) capture | 4/4 | 4/4 | **unconditional** |
| conformal: model intervals beat both baselines (Winkler) | 32/32 cells | | **unconditional** |
| campaign shift: backbone `T_i` beats PCHIP and causal GP | 4/4, 4/4 | 4/4, 4/4 | **unconditional** (one temporal block, 4 inits) |
| `T_i` Δt > 15 ms pooled | PASS | PASS | **unconditional** |
| `V_rot` routing structural / history indispensable | yes | yes | **unconditional** |
| cut threshold 2.5–4 keV | insensitive | (n/a) | reported |
| window model `T_i` beats PCHIP | 3/4 | 4/4 | conditional |
| b3k8 (21k) = backbone `T_i` | +0.002 | −0.194\* | **cut-conditional** (§8z restated) |
| non-peak `T_i`, backbone | 4/4 | 2/4 | conditional |
| `T_i` Δt > 45 ms pooled | PASS | tie | conditional |
| MNAR-reweighted `T_i` vs PCHIP | 2/4 | 4/4 | conditional (vs persistence 4/4 both) |
| `V_rot` vs PCHIP (PR4) | 1/4 | 2/4 | not established; vs persistence 3/4 both, campaign 4/4 both, Δt > 15 ms PASS both |

**Fit-failure spike structure — B.7 follow-up audit memo** (`experiments/b5_rescore/
spike_structure_audit.py` → `data/.b5_spike_structure.json`, 641 files, observed rows in time
order, blocks split at ≥ 0.5 s gaps). The 1,197 `CES_TI` rows above 3 keV form **951 runs in
274 shots**: 85% are single rows (813), 121 runs are 2–4 rows long, only **17 runs (2%) last
≥ 5 rows** (max 15); 70% of runs are *isolated* (a single row with both observed neighbours
< 2 keV) and the median run peak is **13×** the mean of its neighbours (IQR 6–26×). So the cut
removes almost exclusively one-sample events — what a fit failure looks like, not what an
ELM/transient looks like. But single-sample outliers are **bidirectional**: on the 225,580
observed rows with two observed neighbours, 3,845 are ≥ 2× *above* both neighbours and 4,965 are
≤ ½× *below* both (dips), and the 3 keV cut removes only **19%** (731) of the upward outliers
(their median value is ≈ 0.9 keV) and none of the dips. A value cut is therefore partial and
one-sided; it is the pre-registered, physically-motivated proxy ("> 3 keV is not a KSTAR ion
temperature") until CES fit-quality metadata (fit χ², signal level) arrives, at which point a
quality cut replaces or accompanies it (§8v). `CES_VT` (no cut in the protocol): |V_rot| p99 =
223, p99.9 = 666, max 9,725 km/s; **119 rows > 1,000 km/s in 16 shots**, of which **101 are one
sustained block in s31181** and the other 18 are single-row spikes; 145 single-row jumps
> 300 km/s against quiet (|·| ≤ 300) neighbours, 588 > 100 km/s. That is the population §8z's
spiked anchors come from (0–4 rows per TEST split carrying 28–72% of b3's `V_rot` SSE). No
protocol change is made here — and **승상님 decided (2026-08-16) that `V_rot` stays uncut**: no value
or jump rule, no retraining; every persistence-anchored `V_rot` comparison must report the
spike-anchor rows' share of squared error, to be revisited if CES fit-quality metadata arrives.
The same decision kept the two-population report as co-primary (no p100-only headline) and left
B.6 waiting (no kHz Mirnov features in `data/` as of 2026-08-16).

**What this section does not show.** No model was selected or re-selected here (the backbone is
§8x's; b3k8 is §8z's rung). The campaign evidence is one temporal block × 4 init seeds, not four
temporal splits. The MNAR `V_rot` row is uninformative by coverage, not negative. And every p100
`T_i` verdict is hostage to the ≈ 1% spike-anchor rows that carry 70–83% of every arm's squared
error — which is why the cut population remains co-primary rather than a sensitivity row, and
why the p100 population's own headline is not quoted alone.

Artifacts: `data/.b5_summary.json` (verdict), `data/.b5i_{w2,seqv2,anchor,b3k8}_s*`,
`data/.b5_camp_*`, `data/.b5c{2500,4000}_seqv2_s*`, `data/.b5_abl_*`,
`data/.b5_spike_structure.json`, `data/.campaign_split_manifest.json`; the cut-population inputs
are B.1's `.b1_*` and §8z's `.b3c_b3k8_s*` / `.b3_anchor_s*` untouched (val npz added to the
backbone dirs additively, TEST reports split-tagged).

---

## 8ac. Effective recurrent reach + the real-time causal ladder (2026-08-17) — `T_i` uses 500 ms of context, and two steps of it is not enough

**Question (승상님).** Two challenges to the adopted backbone, asked together. (1) Multi-sensor
papers routinely use convolutional stacks rather than an LSTM — what justifies the recurrence
here? (2) The window family is smaller and looked physically adequate (§8f: `T_i` skill plateaus
at `W = 2`), so why carry an unbounded recurrent state at all? Both reduce to one measurable
quantity that had never been measured: **how many contiguous past steps does the trained model
actually use?** §8aa closed the *width* axis; this closes the *reach* axis.

**Design** (`experiments/reach/`, runners `run_reach.py` → `bench_causal_arms.py`; 6 min wall for
the ladder with `--jobs 4`; artifacts `data/.reach_s*`, `data/.reach_summary.json`,
`data/.reach_pareto.json`). **No retraining.** The four frozen B.1 backbone checkpoints
(`.b1_seqv2_s{seed}_i{seed}`, §8x) are re-scored on the *same* TEST population with the recurrent
state reset `ctx` steps before every scored row (`eval_seq._forward_truncated`, reached by the new
additive env `CES_SEQ_CONTEXTS`; unset reproduces the frozen path exactly). Sequence blocks run
median 298 rows = 3.0 s (p95 796, max 1482), and `train_seq.py` applies no truncation, so
full-block inference is the model's true reach.

Only the *recurrent* reach is cut. The per-target carry-forward / staleness INPUT channels are
still computed over the whole block by `seq_data.build_blocks`, because the `W = 2` window control
receives exactly that carried history — so `ctx = 2` is the honest "same information as the window
family" rung, not a crippled model. **Identity check: `ctx = full` reproduces each frozen
`se_model`, `se_pchip`, `shot` and `dt_ms` array bit-identically on 4/4 splits**, which is the
whole warrant for reusing the frozen population — every truncated column then differs for exactly
one reason.

**Headline baseline is `gp_causal`, not PCHIP** (승상님): a reach probe asks how much *past* the
model needs, so the reference must itself be causal, and PCHIP cannot run online at all. The
`persistence` column is carried alongside because it is what the claim actually displaces — on the
10 ms grid, when CES is missing, holding the last value **is** current practice.

### 1. Reach ladder (mean over 4 splits; "worst gap" = largest |paired deficit vs full| on any split)

`CES_TI`:

| ctx (steps × 10 ms) | vs `gp_causal` | vs persistence | vs PCHIP | % of persistence margin | sig. deficit | worst gap |
|---:|---:|---:|---:|---:|---|---:|
| 1 | −0.881 | **−0.283** | −0.623 | −73.0% | 4/4 | 1.259 |
| 2 | −0.340 | +0.086 | −0.157 | 20.7% | 4/4 | 0.627 |
| 3 | −0.112 | +0.242 | +0.041 | 60.6% | 4/4 | 0.315 |
| 5 | +0.013 | +0.327 | +0.149 | 82.6% | 4/4 | 0.136 |
| 10 | +0.087 | +0.378 | +0.213 | 95.5% | 3/4 | 0.050 |
| 20 | +0.100 | +0.387 | +0.224 | 97.7% | 3/4 | 0.041 |
| **50** | +0.113 | +0.396 | +0.235 | **99.9%** | 0/4 | 0.0011 |
| 100 / 300 / full | +0.113 | +0.396 | +0.236 | 100.0% | 0/4 | ≤ 1e-5 |

`CES_VT`:

| ctx | vs `gp_causal` | vs persistence | vs PCHIP | % of persistence margin | sig. deficit | worst gap |
|---:|---:|---:|---:|---:|---|---:|
| 1 | +0.015 | +0.288 | +0.128 | 74.9% | 4/4 | 0.474 |
| 2 | +0.111 | +0.358 | +0.214 | 91.7% | 4/4 | 0.117 |
| 3 | +0.135 | +0.376 | +0.236 | 96.1% | 3/4 | 0.044 |
| 5 | +0.146 | +0.384 | +0.246 | 98.4% | 2/4 | 0.016 |
| 10 | +0.152 | +0.389 | +0.251 | 99.5% | 1/4 | 0.005 |
| **20** | +0.154 | +0.390 | +0.253 | **100.0%** | 1/4 | 0.00008 |
| 50 / 100 / 300 / full | +0.154 | +0.390 | +0.253 | 100.0% | 1/4 | ≤ 8e-5 |

Saturation is read with an effect-size floor (`PRACTICAL_EPS = 0.002` skill) as well as by
significance, because the arms are paired row-for-row: `CES_VT` on split 7 keeps a *statistically*
significant deficit of **−5e-7** out to ctx = 300, which is float-level, not an effect. Under the
floor: **`T_i` saturates at 50 steps = 500 ms; `V_rot` at 20 steps = 200 ms.**

### 2. The real-time ladder (10 ms CES budget; skill vs persistence, mean over the same 4 splits)

Baseline latency is the predictor call on an already-built neighbour set; network latency is the
forward pass only (§8l scope). Feature assembly is excluded for both. **Every row below was
re-measured in one idle session** (CPU, batch 1) rather than quoting §8l's frozen numbers — see
the accuracy note after the table for why that mattered. GP cost is measured on the **real**
neighbour sets of the frozen TEST files (|neighbours| median 544 for `T_i`), since `gp_causal`
refits a Matern-3/2 GP per row with hyperparameters selected by exact log marginal likelihood over
a 5 × 4 grid on up to 16 past neighbours.

| arm | causal? | `T_i` skill | `V_rot` skill | median | p99 | p99 vs budget | deployable |
|---|---|---:|---:|---:|---:|---:|---|
| persistence | yes | 0 (ref) | 0 (ref) | 0.010 ms | 0.034 ms | 0.3% | yes |
| `ar_local` | yes | −4.039 | −1.462 | 0.015 ms | 0.045 ms | 0.4% | yes (but far worse than holding) |
| `gp_causal` | yes | +0.319 | +0.276 | 1.136 ms | 2.344 ms | 23.4% | yes |
| **`seq_v2` (stateful 1-step)** | yes | **+0.396** | **+0.390** | **0.966 ms** | **1.494 ms** | **14.9%** | **yes** |
| window `iter009` `W = 2` | yes | — | — | 2.854 ms | 4.455 ms | 44.6% | yes |
| window `iter009` `W = 4` | yes | — | — | 3.121 ms | 5.053 ms | 50.5% | yes |
| linear (acausal) | no | +0.246 | +0.214 | 0.020 ms | 0.046 ms | — | no (needs future) |
| PCHIP (acausal) | no | +0.209 | +0.182 | 0.333 ms | 0.689 ms | — | no (needs future) |
| GP (acausal) | no | +0.396 | +0.390 | 1.347 ms | 2.562 ms | — | no (needs future) |

**Accuracy note — quote the ordering, not the absolutes.** This table's first draft quoted §8l's
frozen `window_iter009 W = 2` p99 of 18.9 ms and concluded the window family misses the deadline.
Re-measuring in one session refutes that: 4.455 ms, 44.6% of budget, comfortably inside. The frozen
value was already suspect on physical grounds — it put `W = 2` at 2.3× the p99 of `W = 4`, though
`W = 2` walks the shorter sequence — and this session reproduces the sane order (`W = 2` 4.455 <
`W = 4` 5.053). PROJECT_KNOWLEDGE recorded "laptop power states move absolutes up to 2×, never the
ordering"; a 4.2× gap exceeds that, so the frozen tail was contaminated, not merely throttled.
**The claim that survives is the ordering, which both sessions agree on:** `seq_v2` step <
`gp_causal` < window `W = 2` < window `W = 4`. No conclusion in this section rests on an absolute
millisecond value, and the reach ladder (§1 above) is a skill measurement untouched by any of this.

### 3. Why the arithmetic is free and the latency is not (2026-08-17, 승상님)

Two follow-ups — "a `W = 2` model should be smaller, why 201k parameters?" and "it is a CNN, it is
a few multiplies, why is it slow?" — turn out to have the same answer, and it is not the one the
parameter counts suggest. Measured with forward hooks (MAC counts) and the same batch-1 timing
loop:

| | params | MACs / step | leaf ops | median | effective |
|---|---:|---:|---:|---:|---:|
| window `iter009` (`W = 2`) | 201,258 | 274,416 | **57** | 3.56 ms | 154 MFLOP/s |
| `seq_v2` stateful step | 357,570 | 353,920 | **10** | 1.02 ms | 691 MFLOP/s |

- **Parameter count does not depend on the window.** `W = 2` and `W = 4` instantiate the *same*
  201,258 weights — the window changes sequence length, not the convolution kernels, the GRU cell
  or the heads. "`W = 2` should be smaller" does not follow.
- **The arithmetic really is negligible, exactly as the intuition says.** 274k MACs at a modest
  5 GFLOP/s is **0.11 ms — 3% of the measured 3.56 ms.** The other 97% is per-operator dispatch
  at batch 1, and the two models' op counts (57 vs 10) predict their ordering better than their
  parameter counts do: `seq_v2` performs **1.29× more multiplies and is 3.5× faster**.
- **Shrinking *widths* therefore does not buy latency.** Window at quarter widths: 121,050 params
  (−40%) → 2.81 ms (−21%). `seq_v2` at `hidden_ti = 24`: 34,162 params (−90%) → 0.86 ms (−17%).
  The *smallest* window variant (121k) is still **2.7× slower than the largest `seq_v2`** (358k).
  Shrinking the **structure** is a different axis and does buy latency — §8ad measures it, and
  what it costs.
- **Fusing the graph is the lever that works.** `torch.jit.trace` + `freeze` on the window model:
  3.05 → **1.76 ms, 1.73× faster**, outputs identical to 1e-5. Not applied to any scored artifact —
  this is a deployment note, not a change to the measured pipeline.

Read together with §8aa (skill flat from 34k to 879k parameters) and §8z (a 21k latent model
matches the backbone on `T_i`), the capacity of *both* families is far above what this input set
needs; the size axis is closed for cost as it was for skill.

**Verdict.**

1. **The window framing is refuted on information, and is merely more expensive on cost.**
   On information — the decisive axis: `ctx = 2`, the contiguous context a `W = 2` model sees,
   loses to the causal GP by **−0.340** on `T_i` and recovers only 20.7% of the persistence margin,
   while full reach beats it by +0.113. §8f's "`W = 2` is enough" was about how many past *CES
   observations* to staple on; it never measured the dense-diagnostic context, and that is where
   the skill is. On cost, the honest statement is weaker than the first draft's: the window model
   **does** fit the 10 ms budget (p99 4.455 ms, 44.6%), it is simply **3.0× the backbone's tail**
   (1.494 ms, 14.9%), because it re-runs three sensor CNNs over the whole window every step while
   the LSTM carries its summary in O(1). "Lighter in parameters" (201k vs 358k) does not mean
   cheaper to run — but it does not mean undeployable either.
2. **`ctx = 1` is worse than doing nothing** (`T_i` −0.283 vs persistence): the instantaneous
   fast-diagnostic state alone does not identify `T_i`. The recurrence is not decoration.
3. **The backbone is the best deployable arm on both axes at once** — higher skill than
   `gp_causal` (+0.396 vs +0.319 on `T_i`) at **1.6× lower p99** (1.494 vs 2.344 ms), and the
   cheapest tail of every arm that beats persistence. It also *ties the acausal GP*
   (+0.396 / +0.390 vs +0.396 / +0.390) using past data only. This is §8p's "the model ties the GP"
   restated where it matters: the tie now happens against the arm that is allowed to see the future,
   from an arm that is not. Note what is **not** claimed: no causal arm here misses the deadline, so
   the argument for the backbone is skill-per-millisecond, not feasibility.
4. **The `T_i` / `V_rot` asymmetry appears again, in a new coordinate.** `V_rot` needs 200 ms of
   context, `T_i` 500 ms — consistent with §8ab's routing result: `V_rot` rides a highly
   autocorrelated carried value, `T_i` integrates fast-diagnostic history.

**What it does not show.** This measures the reach of *this trained model*, not the reach a
differently-trained model would need. Truncation resets the state to zero, so part of the short-ctx
deficit is warm-up rather than missing information — the two cannot be separated by this design,
though the window control has exactly the same handicap, which is why `ctx = 2` is the fair rung.
Most importantly, **it does not establish that recurrence is the only way to reach 50 steps**: a
dilated causal TCN reaches 63 steps with 5 layers and remains an untested candidate. What the
latency column does add is the reason to expect the recurrence to stay cheaper — an LSTM carries
its summary in O(1) per step, whereas a convolutional stack recomputes its receptive field every
step unless a streaming cache is built explicitly, and the window family's 3.0× tail is that
recomputation being paid over a mere 2 steps; a 50-step receptive field would pay it over 50 unless
cached. That is an argument about cost, not correctness, and it is weaker than the first draft of
this section claimed (see the accuracy note). A TCN arm under the B.2 rule (val-only exploration →
pre-registered decision → TEST once, ≥3/4 significant to promote) is the measurement that would
settle the skill question.

---

## 8ad. W-SLIM (2026-08-17) — sizing the window model to its window is 7.9× smaller and 4.6× faster, and costs `T_i` skill

**Question (승상님).** §8ac §3 answered "why so many parameters?" with "the window never
controlled the count", but the follow-up was sharper: **`W = 2` is a different input, so the
structure itself can be smaller.** That is correct as stated, and §8ac had only measured *width*
shrinkage (which leaves the 57-operator graph intact). This batch derives the structure from the
input and measures what it costs.

**Design** (`experiments/wslim/`, model `model_win_slim.py`, runner `run_wslim.py`; 13.1 min wall
with `--jobs 4`; artifacts `data/.wslim_s*`, verdict `data/.wslim_summary.json`). At `W = 2`
`iter009` spends structure a length-2 sequence cannot use: two stacked `Conv1d(k=3, padding=1)`
per sensor (receptive field 5 over 2 steps, so a third of every kernel reads only zero padding),
an `AdaptiveAvgPool1d(1)` that averages the 2 steps and **discards the ordering the window
existed to provide**, and a bidirectional GRU with a 4-head attention pool over those same 2
steps. W-SLIM replaces each stream with `flatten → Linear → GELU` and **keeps the `V_rot` routing
exactly** (verified structurally: perturbing the fast channels changes `T_i` and leaves `V_rot`
bit-identical). **25,602 params / 21 leaf ops / 0.66 ms median** vs 201,258 / 57 / 3.02 ms.

One controlled variable — the architecture, via `CES_MODEL_FILE`. Everything else is B.1 stage A
verbatim: same `GATE_ENV`, same frozen split manifests (`data/.b1_manifest_s*`, test-isolation
assert passed 4/4), same seeds, same harness; the control is the frozen `.b1_w2cut_s{seed}` run,
paired row-for-row. The reading was fixed before the run.

| seed | params | `T_i` skill vs PCHIP | `T_i` paired vs `w2cut` | `V_rot` skill | `V_rot` paired |
|---|---:|---:|---:|---:|---:|
| 42 | 25,602 | **−0.031** | **−0.086\*** | +0.298 | +0.058 |
| 1 | 25,602 | +0.136 | **−0.084\*** | +0.185 | −0.001 |
| 7 | 25,602 | +0.124 | **−0.106\*** | +0.104 | −0.040 |
| 123 | 25,602 | +0.176 | **−0.072\*** | +0.275 | +0.004 |

(\* = paired CI excludes zero.)

**Verdict — the two targets answer differently, and the pre-fixed rule fires on `T_i`.**

1. **`T_i`: the extra structure earns its keep.** Paired mean **−0.087, 4/4 significant deficits**
   — past the ≥3/4 threshold, so W-SLIM is a *cheaper-but-worse* point, not a free lunch. On split
   42 it drops below PCHIP entirely (−0.031, PR4 fails where the control passed). The intuition
   that a length-2 window needs no structure is wrong for `T_i`.
2. **`V_rot`: the reduction is free.** Paired mean **+0.005, 0/4 deficits and 0/4 wins** — a GRU
   plus a 4-head masked-attention pool over 2 timesteps is fully replaceable by
   `flatten → Linear` here. This is what §8ab's routing result predicts: `V_rot` rides a carried
   value plus staleness, and nothing in that needs sequence machinery.
3. **The cost/skill trade-off is now priced.** 7.9× fewer parameters and 4.6× lower latency for
   −0.087 `T_i`. W-SLIM still beats PCHIP on 3/4 splits, so it remains a legitimate operating
   point if latency ever binds — but nothing in §8ac's ladder says it binds (every causal arm fits
   the 10 ms budget), so **there is no reason to adopt it, and the backbone stays `seq_v2`.**
4. **Capacity is not the mechanism.** §8aa flattened skill across a 26× *width* range and §8z
   compressed the backbone to a 21k latent with no `T_i` loss — yet 25.6k here loses 0.087. So
   what `iter009` buys on `T_i` is **structure, not size**, which is the same lesson §8z drew for
   interpretability, arriving from the opposite direction.

**What it does not show.** Three things changed at once (conv stack → linear, pooling → flatten,
GRU+attention → linear), so the `T_i` deficit is not attributed. The measurement that would
attribute it is a two-arm follow-up on this same control: restore **only** the sensor conv stack,
then restore **only** the history GRU+attention. Given finding 2 (`V_rot` indifferent to the
history machinery) the sensor conv path is the better first guess for where the 0.087 lives.
Also unmeasured: W-SLIM at larger widths — this file fixed `sensor_feature_dim = 48` (half of
`iter009`'s 96), so part of the deficit may be width rather than structure, which the same
follow-up should separate by running W-SLIM at 96.

---

## 8ae. Trained-at-reach vs. truncated (2026-08-19) — at least 84% of §8ac's `T_i` reach deficit was cold start, not missing information

**Question (승상님).** Before designing a model-family × latency-budget comparison (B.9) on top of
§8ac's reach ladder, the ladder itself was re-checked. §8ac drew it by **truncating one trained
backbone** — the recurrent state is reset to zero `ctx` steps before every scored row — and its
Verdict 1 then read the `ctx = 2` rung as an *information* statement ("the window framing is
refuted on information"). §8ac's own "What it does not show" had flagged the gap ("part of the
short-ctx deficit is warm-up rather than missing information — the two cannot be separated by
this design"). It can be separated, and the artifacts to do it were already on disk.

**Design** (`experiments/reach/trained_vs_truncated.py`; seconds of wall, no retraining, no new
scoring; artifact `data/.reach_trained_vs_truncated.json`). `data/.b1_w2cut_s{seed}` is a model
**trained at reach 2** (the B.1 stage-A window control, §8x), scored by the same
`compare_baselines` path on the **same rows** as `data/.reach_s{seed}`. Three points per target
bracket the question, and the script refuses to compute anything until `shot`, `dt_ms`, `y_true`
and `se_persistence` are verified **bit-identical** between the two files on all 4 splits (they
are). Statistic is the same shot-clustered paired bootstrap as every other batch.

### 1. Skill vs. persistence, mean over the 4 splits (identical rows)

| target | full (whole block) | **trained@2** | trunc@2 (§8ac rung) | truncation deficit | recovered by training | warm-up share | full − trained@2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `CES_TI` | +0.396 | **+0.346** | +0.086 | 0.310 | **+0.260** | **84%** | +0.049 |
| `CES_VT` | +0.390 | **+0.357** | +0.358 | 0.032 | −0.001 | −3% | +0.033 |

Paired bootstrap, per split (skill of the first arm relative to the second; \* = CI excludes 0):

| | s42 | s1 | s7 | s123 | sig. |
|---|---:|---:|---:|---:|---|
| `CES_TI` trained@2 vs trunc@2 | +0.294\* | +0.260\* | +0.313\* | +0.268\* | **4/4** |
| `CES_TI` full vs trained@2 | +0.130\* | +0.058\* | +0.062 | +0.044 | 2/4 |
| `CES_VT` trained@2 vs trunc@2 | −0.092\* | +0.060\* | +0.027\* | −0.027\* | sign split |
| `CES_VT` full vs trained@2 | +0.181\* | −0.003 | −0.004 | +0.045\* | 2/4 |

**These 4 runs are the diagonal (`init = split`) of §8x's 16-run grid, not a new gate.** §8x's
properly-powered estimate of the same *full vs trained@2* contrast is pooled **+0.081, CI
[+0.067, +0.096], 13/16 individually significant** — the 2/4 here is the expected subset behaviour
of 4 draws from that grid (per-split init-means were +0.129 / +0.059 / +0.078 / +0.058), **not** a
challenge to §8x. The backbone still beats the window control.

### 2. `CES_VT` is the internal control, and it behaves as the routing result predicts

§8ab established that the `V_rot` branch never reads the fast diagnostics and rides carried input
channels that `seq_data.build_blocks` computes over the whole block **regardless of truncation**.
So cutting the recurrent state should cost `V_rot` almost nothing — and it costs 0.032, one tenth
of `T_i`'s 0.310, with training at that reach recovering none of it (−0.001). The warm-up reading
of the `CES_TI` gap therefore is not a convenient story: the same truncation applied to the branch
that does not use its recurrent state produces no such gap.

**Verdict.**

1. **§8ac's `ctx = 2` rung is not an information measurement, and Verdict 1 of §8ac is hereby
   narrowed.** A model *trained* at reach 2 recovers +0.260 of the 0.310 the truncated backbone
   lost on `T_i`, 4/4 significant. Truncation resets a state the model was never trained to
   rebuild in 2 steps; that is a warm-up penalty, not an absence of information.
2. **What full-block context is worth on `T_i` is bounded above by +0.081, not +0.310.** The
   `full − trained@2` contrast is the *combined* reach + architecture gap (`seq_v2` whole block vs
   window `iter009` at `W = 2`), and §8x prices that combination at +0.081 pooled. Pure reach is
   therefore worth **at most** +0.081 and §8ac's ladder overstates it by **≥ 3.8×**.
3. **The `T_i` skill does not come mainly from long memory.** A model trained at 20 ms of context
   already reaches **87.5%** of the backbone's margin over persistence. Whatever the backbone's
   remaining advantage is, §8ad (structure, not size) and §8ab (routing) are the better candidates
   than reach — and B.9 is designed to separate them.
4. **`ctx = 1` is unaffected as a claim about this model** (§8ac Verdict 2), but the same objection
   applies to reading it as an information statement, and B.9's ladder re-measures it.

**What it does not show.** `trained@2` here is a different **architecture** (window `iter009`), not
`seq_v2` retrained with a 2-step state, so reach and structure are confounded in every number in
the right-hand column. This bounds the warm-up share rather than measuring it: the pure-reach
value is **unmeasured**, and the measurement that would settle it is `seq_v2` trained with the
recurrent state reset every `ctx` steps — pre-registered as B.9's reach ladder
(`experiments/PREREGISTRATION_B9.md`). Until that runs, quote this as "a model trained at this
reach reaches +0.346", never as "`seq_v2` at reach 2 reaches +0.346".

---

## 8af. B.9 axis A (2026-08-19) — trained at each reach, `T_i` and `V_rot` both saturate at **70 ms**, not 500

> **Superseded on the number by §8al (2026-08-20).** Rungs 4, 5, 6 and 10 were added.
> The pre-registered §3.4 rule now returns **50 ms**, not 70 — 7 was the first rung *below*
> the bar, not the smallest, because 4/5/6 had never been trained. Everything else here stands.

**Question (승상님).** §8ae bounded the warm-up share using the `W = 2` window control, a
*different architecture*, so reach and structure stayed confounded and the pure-reach value was
unmeasured. This measures it: `seq_v2`, byte-identical at every rung, **trained and scored** at
reach 2 / 7 / 15 / 31 / 63 (`CES_SEQ_TRAIN_CTX` / `CES_SEQ_EVAL_CTX`). Pre-registered before any
number existed in `experiments/PREREGISTRATION_B9.md` (axis A, H1–H2).

**Design** (`experiments/b9_reach/run_b9_reach.py`, 20 runs, artifacts `data/.b9_v2r*_s*`, verdict
`data/.b9_reach_ladder.json`). Training chops each block into non-overlapping `r`-step segments,
so the recurrent state is never carried further than `r` and every row stays supervised; eval uses
the sliding `_forward_truncated(r)`, so every scored row receives exactly `r` steps. Everything
else is `GATE_ENV` (W = 2 · held-free · 3 keV cut · per-shot norm · frozen B.1 manifests · 4
splits, init = split). **`r = full` is the frozen B.1 backbone, reused, not retrained.**

**Rows per batch is held constant, not blocks per batch** — chunking a median-298-row block at
`r = 2` makes 149 sequences out of one, so a fixed `CES_SEQ_BATCH = 16` would have moved the
effective batch from ~4,800 rows to 32 and changed gradient noise by two orders of magnitude: a
second variable larger than the one under test. Block-batch is derived as `max(8, 4800 / r)`.

Rungs are 2 / 7 / 15 / 31 / 63 so a dilated causal TCN's receptive field (`2^(L+1) − 1`) lands on
the same integers and axis B compares against them directly rather than against an interpolation.

### 1. The two ladders on the same rungs (`CES_TI`, paired vs full-block, mean of 4 splits)

`experiments/b9_reach/truncated_at_rungs.py` re-scores the **same frozen backbones** at these rungs
into `.b9trunc_s*`, leaving §8ac's artifacts untouched; `ctx = full` reproduces each frozen run
**bit-identically on 4/4 splits**, and the rung-2 value **exactly reproduces §8ac's** −0.510.

| rung (context) | truncated (state cut) | sig. | **trained at that reach** | sig. | warm-up share |
|---:|---:|---|---:|---|---:|
| 2 (20 ms) | **−0.510** | 4/4 | **−0.065** | 4/4 | **87%** |
| 7 (70 ms) | −0.064 | 4/4 | +0.002 | 0/4 | 103% |
| 15 (150 ms) | −0.018 | 2/4 | +0.005 | 1/4 | — |
| 31 (310 ms) | −0.009 | 1/4 | +0.023 | 0/4 | — |
| 63 (630 ms) | +0.000 | 0/4 | +0.029 | 0/4 | — |

`CES_VT`: truncated −0.055 (4/4) → trained −0.013 (2/4) at rung 2, warm-up share 76%; both curves
are at zero from rung 7 on.

Skill vs persistence for the trained rungs: `CES_TI` +0.356 / +0.397 / +0.398 / +0.409 / +0.413
and `CES_VT` +0.384 / +0.393 / +0.376 / +0.393 / +0.390 (full-block backbone: +0.396 / +0.390).

### 2. Verdict

1. **Saturation is 7 steps = 70 ms for BOTH targets** (§3.4 rule: the smallest rung with
   |paired| < 0.02 and ≤ 1/4 significant deficits). §8ac's truncation-based values were 50 steps
   (500 ms) for `T_i` and 20 (200 ms) for `V_rot` — **7× and 3× too long**. The truncated ladder
   needs **63 steps to match itself** where a model trained at the reach needs **7**; the factor
   of 9 is entirely cold start.
2. **§8ac measured correctly and was read wrongly.** The independent re-score reproduces its
   ctx = 2 number to the digit. What does not follow from that number is the *information*
   statement §8ac's Verdict 1 made, and §8ae/§8af replace it.
3. **Unbounded context is not needed, and is not better.** Rungs 31 and 63 sit *nominally above*
   the full-block backbone (+0.023, +0.029, 0/4 significant either way). Carrying state across a
   1,482-step block buys nothing over bounding it at 310 ms.
4. **`V_rot` is not reach-free.** At rung 2 it loses 2/4 significant — small (−0.013) but not the
   clean independence §8ab's routing result made plausible. It needs the same 70 ms.
5. **§8x's +0.081 backbone-over-window gap now decomposes**: reach −0.065, architecture ≈ −0.016.
   Four fifths of what the backbone buys over the window family is *how far back it looks*, and
   one fifth is *what it is*.

### 3. Pre-registered hypotheses, executed as written

- **H1 (`T_i` depends weakly on reach: deficit ≤ 0.10 at reach 2) is REJECTED**, on the
  significance clause: the deficit is −0.065 (inside the 0.10 bound) but 4/4 significant, and the
  rule falsified H1 on `평균 > 0.10 **또는** 4/4 유의`.
- **The rule was mis-specified, and is reported that way rather than repaired after the fact.**
  Bundling an effect-size bound and a significance count into one disjunction makes any small but
  robust effect falsify a "weak dependence" hypothesis. The attached consequence — "§8ac Verdict 1
  is restored" — **does not follow**: Verdict 1 asserted −0.34 vs the causal GP from a −0.510
  truncation rung, and the trained measurement is −0.065. The batch's own §3.2 decision rule gives
  the usable reading: |−0.065| ≥ 0.02 with ≥ 3/4 significant = "differs", magnitude 0.065.
  PREREGISTRATION_B9.md is left unedited; the defect is recorded here.
- **H2 (`V_rot` reach-independent) is NOT supported** at rung 2 (2/4 significant, so neither tie
  nor difference under §3.2); it holds from rung 7 on.

**What it does not show.** Truncated training gives rows near a segment boundary less than `r`
steps, while eval gives every row exactly `r` — in-distribution, since the model sees every context
length up to `r` during training, but not identical. The rungs are coarse (2 → 7), so the true
minimum for `T_i` lies somewhere in 3–7 steps and is not resolved here. And this is one
architecture: whether 70 ms is a property of the *plasma* or of `seq_v2` is what axis B tests.

---

## 8ag. B.9 axis B (2026-08-19) — at matched reach the sequence-operator **family does not decide skill**

> **Bounded below by §8ak (2026-08-20).** Every arm here is trained at reach 15 or 63. At the
> threshold rung itself (7 = 70 ms) the attention arm **differs** from the same-reach LSTM;
> the convolutional arm still ties. Read "family does not decide skill" as holding at
> 150 ms and above.

**Question.** §8ac left one architecture question explicitly open: "it does not establish that
recurrence is the only way to reach 50 steps: a dilated causal TCN reaches 63 steps with 5 layers
and remains an untested candidate." §8af then showed the reach it was arguing from is 70 ms, not
500. So the sharp form is: **hold the reach fixed and vary only the operator.**

**Design** (`experiments/b9_family/run_b9_family.py`, 12 runs, artifacts
`data/.b9_{tcn15,tcn63,xfmr63}_s*`, verdict `data/.b9_family.json`). Three families, all carrying
seq_v2's routing (`V_rot` never sees the fast diagnostics), all trained and scored at the reach
their receptive field declares, all under `GATE_ENV` on the frozen B.1 manifests:

- `tcn15` / `tcn63` — dilated causal conv, kernel 3, dilations 1·2·4…, `RF = 2^(L+1) − 1`;
- `xfmr63` — causal banded attention; the per-layer band is *derived* from the target reach because
  two stacked 63-wide bands compose to 125 (caught by `test_seq_family_is_causal_and_reach_bounded`,
  which asserts causality and the declared receptive field numerically);
- position is encoded relatively (a shared ALiBi slope), so a KV cache is valid — §8ah says why
  that mattered.

**Each arm is paired against the axis A rung trained at its own reach**, not only against the
backbone: against the backbone alone, family and reach would be confounded exactly as §8ae showed
they were.

### 1. Paired vs the same-reach LSTM (mean of 4 splits; wins/losses = CI excludes 0)

| arm | RF | params | `CES_TI` | sig. w/l | `CES_VT` | sig. w/l | vs full backbone | §3.2 verdict |
|---|---:|---:|---:|---|---:|---|---:|---|
| `tcn15` | 15 | 184,626 | **+0.014** | 2 / 0 | +0.000 | 0 / 0 | +0.019 | undecided |
| `tcn63` | 63 | 297,810 | **−0.016** | 0 / 2 | −0.036 | 0 / 1 | +0.013 | undecided |
| `xfmr63` | 63 | 295,746 | **−0.019** | 0 / 1 | −0.024 | 0 / 1 | +0.010 | tie |

### 2. Verdict

1. **H3 is not falsified and H4 is supported.** Every arm lands **inside ±0.02** of the LSTM
   trained at the same reach — under the pre-registered practical floor — and **no arm reaches
   3/4 significance in either direction**, so no promotion is triggered and the backbone is
   unchanged. Recurrence, dilated convolution and banded attention are indistinguishable on skill
   once they see the same **150 ms or more** — amended 2026-08-20: this section's arms were
   trained at 150 and 630 ms, so it never observed the 70 ms rung, and §8ak finds the attention
   arm **0.023 below** the LSTM there with 3/4 significance. The convolutional arm ties at 70 ms
   and at 30 ms.
2. **Read with §8af, this is the batch's central result.** Changing the *family* moves `T_i` by
   ≤ 0.019 and never significantly; changing the *reach* from 70 ms to 20 ms moves it by −0.065,
   4/4 significant. **The ranking is set by how far back the model looks, not by what the model
   is** — a statement about the problem, not about the architecture zoo.
3. **§8ac's open TCN question is answered, and the answer removes that argument's premise.** A
   63-step TCN does match the recurrent backbone, so recurrence was never load-bearing for
   *skill*. What it still buys is cost, and §8ah prices it.
4. **All three families sit nominally above the full-block backbone** (+0.019 / +0.013 / +0.010),
   the same direction as §8af's rungs 31 and 63. Bounding the context is, if anything, mildly
   helpful.

**What it does not show.** Three families is not "all families"; a state-space model (S4/Mamba)
would test whether the tie extends to an operator with O(1) state *and* long reach. `tcn15`'s 2/4
wins and `tcn63`'s 2/4 losses are individually real but cancel across splits, which is what the
±0.02 floor exists to express — a larger seed grid could resolve them into a small real ordering,
and that ordering would still be an order of magnitude below the reach effect.

---

## 8ah. B.9 axis C (2026-08-19) — 10 ms never binds; at 1 ms the backbone passes once the **implementation** stops being the cost

> **Partly suspended by §8aj (2026-08-19).** A re-run of the same 5-session protocol recorded a
> **21.84×** session-to-session p99 spread and overwrote the per-session artifacts behind the table
> below, so **the 1 ms column and verdict 2 must not be quoted**. Verdicts 1, 3 and 4 stand; §8aj
> re-derives them from operator counts, which do not depend on the machine.

**Question (승상님).** "10 ms 구간에서는 어느 것이 가장 좋은지 … 1 ms를 새로운 제한으로 주면
그때는 어떤 모델이 최선인지." §8ac priced the arms against 10 ms and found the budget never binds,
so its argument was skill-per-millisecond rather than feasibility. A 1 ms deadline (a control-cycle
constraint rather than a diagnostic-cadence one) is the first place where structure could decide
admissibility.

**Design** (`experiments/b9_latency/bench_budget.py`, artifacts `data/.b9_latency_s{1..5}.json`,
verdict `data/.b9_latency.json`). §8ac concluded "quote the ordering, not the absolutes" after two
sessions disagreed by 4.2× — but a 1 ms verdict *is* an absolute, so PREREGISTRATION_B9.md §4
replaced that rule: **5 independent sessions in separate processes**, 200 warm-up + 2,000 timed
iterations, single-threaded batch-1 CPU, and an arm clears a budget only when **every** session's
p99 does; a max p99 in **[0.8 B, 1.25 B]** is reported as **boundary**, not resolved.
**Measured session-to-session p99 spread: 1.32×** — the protocol held.

Every arm is timed in the form it would be deployed in. Crucially the TCN and the transformer were
given **streaming caches** (per-layer ring buffers; a KV cache) before being timed: §8ac's cost
argument was that a convolutional stack "recomputes its receptive field every step unless a
streaming cache is built explicitly", and pricing them without one would have confirmed that
argument by construction. Both caches reproduce the batch forward to **3e-7**
(`test_seq_family_streaming_equals_batch`).

### 1. Latency ladder (5 sessions, CPU, batch 1, 1 thread)

| arm | mode | params | median | max p99 | 10 ms | **1 ms** |
|---|---|---:|---:|---:|---|---|
| persistence | per-row baseline | 0 | 0.007 | 0.018 | pass | **pass** |
| `gp_causal` | per-row refit | 0 | 0.638 | 2.277 | pass | fail |
| `seq_v2` | stock `nn.LSTM` step | 357,570 | 0.487 | 1.829 | pass | fail |
| `v2m7k` | stock `nn.LSTM` step | 6,866 | 0.291 | 0.804 | pass | boundary |
| `v2m2k` | stock `nn.LSTM` step | 2,362 | 0.331 | 1.153 | pass | boundary |
| `tcn15` | stock streaming cache | 184,626 | 1.100 | 2.522 | pass | fail |
| `tcn63` | stock streaming cache | 297,810 | 1.978 | 4.123 | pass | fail |
| `xfmr63` | stock streaming cache | 295,746 | 1.121 | 2.597 | pass | fail |
| window `W = 2` | recomputed every step | 201,258 | 1.311 | 3.181 | pass | fail |
| `seq_v2` + jit fuse | fused LSTM | 357,570 | 0.440 | 1.602 | pass | fail |
| `v2m7k` + jit fuse | fused LSTM | 6,866 | 0.202 | 0.578 | pass | **pass** |
| `v2m2k` + jit fuse | fused LSTM | 2,362 | 0.207 | 0.485 | pass | **pass** |
| **`seq_v2` lean step** | explicit ops | 357,570 | **0.218** | **0.591** | pass | **pass** |
| **`v2m7k` lean step** | explicit ops | 6,866 | 0.145 | 0.396 | pass | **pass** |
| **`v2m2k` lean step** | explicit ops | 2,362 | 0.140 | 0.539 | pass | **pass** |
| **`tcn15` lean step** | explicit ops | 184,626 | 0.273 | 0.975 | pass | boundary |
| **`tcn63` lean step** | explicit ops | 297,810 | 0.401 | 1.122 | pass | boundary |
| **`xfmr63` lean step** | explicit ops | 295,746 | 0.876 | 2.102 | pass | fail |

Session-to-session p99 spread on this pass: **2.56×** — larger than the fused-only pass (1.32×)
but still below the 4.2× §8ac saw on a contaminated session, and the max-p99 rule absorbs it.

**Why the lean rows exist, and why every family has one.** The first pass put every model arm at
or above 1 ms, so §8j's rule applies: name the measurement that would overturn the negative, then
run it. Profiling the online step named it immediately — the budget was going to module call
protocol, not to the network:

| operation | cost | arithmetic |
|---|---:|---|
| `torch.mm`, 32 × 32 | **1.5 µs** | 32k MAC |
| `nn.LSTM`, 1 layer, hidden 8 | **70.1 µs** | ~1k MAC |
| `nn.LSTMCell`, hidden 8 | 25.7 µs | ~1k MAC |
| `nn.Conv1d`, 128 channels, k = 3 | **214.8 µs** | — |
| `nn.MultiheadAttention`, d = 128 | 108.8 µs | — |

`experiments/b9_latency/lean_steps.py` re-expresses each step in explicit `addmm` / `bmm` calls
over the **same parameters** — a kernel-3 dilated convolution over a ring buffer *is* three matrix
multiplies, and one attention query against a band is a packed QKV projection plus two `bmm`. Each
lean arm is asserted equal to its own model's forward (measured **2e-7**) before it is timed, and
so is the fused arm (1e-5), because `torch.jit.trace` would otherwise bake the first step's
recurrent state into the graph. **All three families get the treatment on purpose**: optimising
only the recurrent arm would have manufactured the conclusion this section previously drew.
Neither fusion nor the lean rewrite is a scored artifact (PREREGISTRATION_B9.md §2.3), and the
pass rule is unchanged — what changed is the implementation being measured, not the decision rule.

### 2. Verdict

1. **At 10 ms nothing binds — every arm passes, by 2× or more.** §8ac reproduced. Structure does
   not decide deployability at the CES cadence, so at this budget the choice is made on skill, and
   §8ag says skill does not distinguish the families. **The 10 ms answer is therefore: take the
   cheapest arm that reaches 70 ms of context**, which is the stateful recurrent step.
2. **At 1 ms the backbone itself passes — the model was never the constraint.** `seq_v2` with a
   lean step clears the deadline at **max p99 0.591 ms** on all five sessions, and so do the
   6,866- and 2,362-parameter variants (0.396 / 0.539 ms). **No shrinking was required.** The
   first pass's "every model arm is at or above 1 ms" was a statement about `nn.Module` call
   overhead, and it did not survive removing it.
3. **Size is not the lever, and the numbers say so twice.** Through stock modules a
   2,362-parameter model and a 357,570-parameter one differ by 1.5× in median; through lean steps
   they differ by 1.6× (0.140 vs 0.218 ms) across a **151× parameter range**, because the cost is
   per-step *operator count*, not parameters or arithmetic. `v2m2k` does ≈ 2.4k MAC per step —
   microseconds of arithmetic — and everything above that is dispatch. This is §8ac §3's finding
   (274k MAC = 3% of a 3.56 ms step) arriving at its extreme.
4. **Skill does not depend on the family (§8ag); price does — but far less than the first pass
   said.** Through stock modules the TCN looked 2.6–4.5× the recurrent step. Given each family the
   *same* lean treatment, that falls to **1.3–1.8×** (0.273 / 0.401 vs 0.218 ms median): most of
   the gap was `nn.Conv1d`, not convolution. Attention stays genuinely expensive (0.876 ms, 4.0×).
   **At 10 ms this ordering is irrelevant — everything passes. At 1 ms it decides the answer:
   recurrent passes, convolutional lands on the boundary (0.975 / 1.122 ms), attention fails
   (2.102 ms).** So the budget, not the physics and not the skill, is what selects the backbone.
5. **The honest correction.** This section's first version concluded "no arm resolves at 1 ms" and
   "recurrence wins on price". Both were artifacts of measuring stock implementations: the first is
   now false, and the second is true by a factor of 1.5 rather than 4. The decision rule was never
   changed — only the implementation under it.

**What it does not show, and the measurement that would settle it** (§8j rule). One machine, one
runtime, CPU, and a *hand-written* step rather than a compiled one — so these are an upper bound
on cost, not a floor. The remaining named measurements are (a) a compiled runtime (ONNX Runtime, a
quantized kernel) under this same 5-session protocol, which should take the recurrent arm well
below 0.1 ms and could move both convolutional arms off the boundary, and (b) the control system's
own hardware, since every absolute here is machine-specific. Note also what the boundary arms
being *close* means: `tcn15`/`tcn63` sit at 0.975 / 1.122 ms and would very likely pass compiled,
so "convolution cannot make 1 ms" is **not** a claim this section supports.

---

## 8ai. B.9 axis D (2026-08-19) — the family tie **breaks below 10k**: convolution reaches the same ceiling with half the parameters

**Question.** §8ag tied the three sequence-operator families at ~300k parameters once the reach
matched. `b8_minimal` swept 1k–10k, but with **recurrent arms only** (`v2m*` / `b3m*`), so the band
where an inductive bias should matter most had never been asked the family question. Pre-registered
as PREREGISTRATION_B9.md §2.3-D — **H6: the family tie holds down there too** (|paired| < 0.02,
significance ≤ 1/4 against the size-matched recurrent arm).

**Design** (`experiments/b9_minimal/run_b9_minimal.py`, 12 runs, artifacts
`data/.b9_{tcn8k,tcn3k,tcn2k}_s*`, verdict `data/.b9_minimal_family.json`). Reach is fixed at
**15** for every arm — above the 7-step saturation §8af measured and equal to a 3-layer TCN's
receptive field — so the operator is the only variable. Each arm is paired three ways: against the
size-matched recurrent arm from `b8_minimal`, against `v2r15` (the same reach at full width, from
axis A), and against the 357k backbone. `GATE_ENV`, cut population, the frozen B.1 manifests.

**This is no longer a deployment argument.** §8ah found that with a minimal-operator step even the
357k backbone is far from being the cost, so small models are not required by any budget. What is
left is the measurement itself: how far down does "the family does not matter" hold.

### 1. Paired vs the **size-matched recurrent** arm (mean of 4 splits; w/l = CI excludes 0)

| arm | params | recurrent match | params | `CES_TI` | sig. w/l | `CES_VT` | sig. w/l | §3.2 verdict |
|---|---:|---|---:|---:|---|---:|---|---|
| `tcn8k` | 8,094 | `v2m7k` (LSTM) | 6,866 | **+0.028** | 4 / 0 | −0.020 | 0 / 1 | **differs** |
| `tcn3k` | 3,238 | `v2m2k` (LSTM) | 2,362 | **+0.040** | 4 / 0 | −0.010 | 0 / 1 | **differs** |
| `tcn2k` | 1,808 | `b3m1k` (B.3 GRU) | 1,208 | **+0.027** | 3 / 0 | +0.303 | 2 / 0 | **differs** |

### 2. The same arms against the full-width rungs — where the tie *does* hold

| arm | params | vs `v2r15` (184,626) | sig. | vs backbone (357,570) | sig. | vs `W = 2` window | skill vs causal GP | pass |
|---|---:|---:|---|---:|---|---:|---:|---|
| `tcn8k` | 8,094 | +0.007 | 0 / 0 | +0.012 | 1 / 0 | +0.084 | +0.124 | **4/4** |
| `tcn3k` | 3,238 | +0.001 | 0 / 0 | +0.006 | 1 / 0 | +0.079 | +0.119 | **4/4** |
| `tcn2k` | 1,808 | −0.004 | 0 / 0 | +0.001 | 1 / 0 | +0.074 | +0.114 | **4/4** |

(`CES_TI`; the causal-GP column is the absolute skill the arm scores, and "pass" counts splits where
its CI clears zero.)

### 3. Verdict

1. **H6 is falsified.** All three convolutional arms beat their size-matched recurrent arm on
   `CES_TI` by **+0.027 … +0.040** with **3–4/4** significance — above `PRACTICAL_EPS` and at the
   significance §3.2 demands in the same direction. Where capacity is scarce, the inductive bias
   pays, and the pre-registered consequence applies: **the convolutional arm is the default shape
   for a minimal model**, not the recurrent one.
2. **This bounds §8ag rather than contradicting it.** Against the *full-width* arm at the same
   reach every TCN is within **±0.007**, and within **±0.012** of the 357k backbone, at 1/100 of
   its parameters. So the two results compose into one statement: **the family does not decide the
   ceiling, it decides how many parameters you need to reach it.** At ~300k both families are at
   the ceiling and the operator cannot show; below 10k the ceiling binds and it does.
3. **The parameter floor that keeps 4/4 against the causal GP roughly halves.** `b8_minimal`'s
   recurrent floor was `v2m4k` at **3,898** params (`v2m2k` at 2,362 fell to 3/4, `b3m1k` at 1,208
   to 3/4). `tcn2k` holds **4/4 at 1,808** — 2.2× smaller — and its mean paired gap to the backbone
   is **+0.001 with zero significant losses**, where `v2m2k` was −0.036 with 3. A 1,808-parameter
   model that beats the causal GP on all four splits is the smallest thing this project has that
   still carries the §8af claim.
4. **The `CES_VT` column supports nothing here, and the one large number is a confound.** Against
   the two LSTM arms `V_rot` is −0.020 / −0.010 and never 3/4 significant. The +0.303 is against
   `b3m1k`, which is the B.3 *interpretable* family with a single-latent `V_rot` branch — that
   contrast varies family, size and branch structure at once, so it is reported and not used.

**What it does not show, and the measurement that would settle it** (§8j rule). Reach is fixed at
15 and depth at 3 layers, so this samples three points on the convolutional ladder rather than
locating its floor; and the recurrent comparands come from `b8_minimal`'s own ladder, not from a
matched sweep. The named measurement is **the TCN width sweep `b8_minimal` did for the LSTM** —
same protocol, widths chosen to land on 1k / 2k / 4k / 8k — which would say where the convolutional
floor actually is, rather than that it is below 1,808.

---

## 8aj. B.9 axis C, second pass (2026-08-19) — cost is **dispatched operator count**, and this machine can no longer decide 1 ms

**Question.** §8ah's 1 ms verdict is decided by an absolute, and absolutes are the one thing this
machine has never delivered reliably. Re-running the pre-registered 5-session protocol — after
axis D made the small convolutional arms candidates and after `tight_step.py` added an
operator-fused `seq_v2` step — asks whether that verdict reproduces.

**It does not, and the protocol is what says so.**

**Design.** `experiments/b9_latency/bench_budget.py --all` (5 sessions, 200 warm-up + 2,000 timed
iterations, batch 1, 1 thread, CPU), artifacts `data/.b9_latency_s{1..5}.json`, verdict
`data/.b9_latency.json`, minimum-of-sessions `data/.b9_latency_min.json`. New in this pass:
`tight_step.py` (four exact fusions of the `seq_v2` step — both branches' first layer in one
matmul, input and recurrent weights in one matmul, gates reordered so each activation is one
contiguous slice, and both branches' cell update at once), the axis D convolutional arms, and
`experiments/b9_latency/op_count.py`, which counts what the milliseconds were always proxying.

### 1. The 1 ms verdict is **suspended**, on this machine, by the rule that was written for it

Worst session-to-session p99 spread on this pass: **21.84×** — against 4.2× in §8ac and 2.56× on
the pass §8ah recorded. Under PREREGISTRATION_B9.md §4 (an arm clears a budget only when *every*
session's p99 does), **not one model arm clears 1 ms in this pass**, including the arms §8ah
recorded as clearing it at max p99 0.4–0.6 ms.

That is not a new physical finding. It is the pre-registered rule correctly refusing to resolve an
absolute from a contaminated sample, which is exactly the failure mode §4 was written to catch.
The consequence is bookkeeping, and it is stated plainly:

- **§8ah's 1 ms table and its verdict 2 ("at 1 ms the backbone itself passes") are suspended, not
  replaced.** The per-session artifacts behind them were overwritten by this pass, so they can no
  longer be checked, and neither table should be quoted as a 1 ms result in the thesis.
- **§8ah's verdicts 1, 3, 4 stand**, because they are *ratios and orderings* — 10 ms never binds;
  size is not the lever; most of the family gap was `nn.Conv1d`, not convolution — and this section
  re-derives all three from statistics that survive additive noise.

### 2. The ladder read on the **minimum** of five sessions

Contention adds time and never removes it, so the minimum over independent sessions is the closest
this pass gets to the machine's own cost. It is therefore a **lower bound**: it is the right
statistic for comparing arms and the wrong one for granting a pass, and no pass verdict is drawn
from it below.

| arm | mode | params | min median (ms) | min p99 | max p99 | session spread |
|---|---|---:|---:|---:|---:|---:|
| `persistence` | per-row baseline | 0 | 0.010 | 0.022 | 0.295 | 13.4x |
| `gp_causal` | per-row baseline | 0 | 1.148 | 2.522 | 16.713 | 6.6x |
| `seq_v2` | stock `nn.LSTM` step | 357,570 | 0.793 | 1.667 | 15.130 | 9.1x |
| `v2m7k` | stock `nn.LSTM` step | 6,866 | 0.389 | 1.095 | 4.525 | 4.1x |
| `v2m4k` | stock `nn.LSTM` step | 3,898 | 0.544 | 1.069 | 3.977 | 3.7x |
| `v2m2k` | stock `nn.LSTM` step | 2,362 | 0.509 | 1.114 | 3.896 | 3.5x |
| `tcn15` | stock streaming cache | 184,626 | 2.074 | 3.832 | 13.247 | 3.5x |
| `tcn63` | stock streaming cache | 297,810 | 2.248 | 5.553 | 40.799 | 7.3x |
| `xfmr63` | stock streaming cache | 295,746 | 1.056 | 2.156 | 32.375 | 15.0x |
| `window_w2` | recomputed every step | 201,258 | 2.418 | 4.370 | 30.640 | 7.0x |
| `seq_v2_jit` | jit-fused LSTM | 357,570 | 0.688 | 1.292 | 5.652 | 4.4x |
| `v2m7k_jit` | jit-fused LSTM | 6,866 | 0.383 | 0.726 | 6.069 | 8.4x |
| `v2m2k_jit` | jit-fused LSTM | 2,362 | 0.379 | 0.726 | 3.691 | 5.1x |
| **`seq_v2_lean`** | lean explicit ops | 357,570 | **0.378** | 0.834 | 7.072 | 8.5x |
| `v2m7k_lean` | lean explicit ops | 6,866 | 0.283 | 0.644 | 14.058 | 21.8x |
| `v2m4k_lean` | lean explicit ops | 3,898 | 0.189 | 0.580 | 2.944 | 5.1x |
| `v2m2k_lean` | lean explicit ops | 2,362 | 0.269 | 0.580 | 5.995 | 10.3x |
| `tcn15_lean` | lean explicit ops | 184,626 | 0.488 | 1.196 | 2.387 | 2.0x |
| `tcn8k_lean` | lean explicit ops | 8,094 | 0.441 | 0.959 | 4.456 | 4.6x |
| `tcn3k_lean` | lean explicit ops | 3,238 | 0.453 | 1.007 | 6.747 | 6.7x |
| `tcn2k_lean` | lean explicit ops | 1,808 | 0.522 | 1.056 | 6.082 | 5.8x |
| `tcn63_lean` | lean explicit ops | 297,810 | 0.798 | 1.787 | 5.807 | 3.2x |
| `xfmr63_lean` | lean explicit ops | 295,746 | 1.514 | 2.866 | 8.562 | 3.0x |
| `xfmr15_lean` | lean explicit ops | 295,746 | 1.737 | 2.961 | 7.099 | 2.4x |
| **`seq_v2_tight`** | operator-fused step | 357,570 | **0.352** | 0.751 | 1.914 | 2.6x |
| `v2m4k_tight` | operator-fused step | 3,898 | 0.226 | 0.533 | 1.683 | 3.2x |
| `v2m2k_tight` | operator-fused step | 2,362 | 0.241 | 0.568 | 1.434 | 2.5x |
| `xfmr63_tight` | operator-fused step | 295,746 | 1.504 | 2.692 | 8.075 | 3.0x |
| `xfmr15_tight` | operator-fused step | 295,746 | 1.273 | 2.566 | 11.095 | 4.3x |

The implementation ladder for the backbone reads **0.793 → 0.688 → 0.378 → 0.352 ms** (stock →
jit-fused → lean → operator-fused): **2.25× total**, of which the lean rewrite is 2.10× and the
four exact fusions add a further 1.07×. Fusion has almost run out; §3 says why.

### 3. Operator counts — the half of cost that does not depend on the machine

`op_count.py` executes one online step under `torch.profiler` and tallies every dispatched `aten::`
operator. The count is exact, reproducible anywhere, and — unlike every millisecond above — did not
move when the machine was busy (`data/.b9_op_counts.json`).

| arm | mode | reach | `aten::` ops / step | min median (µs) | µs per op |
|---|---|---:|---:|---:|---:|
| `seq_v2` | stock `nn.LSTM` | any | 118 | 793 | **6.72** |
| `seq_v2_lean` | lean | any | 161 | 378 | 2.35 |
| `seq_v2_tight` | fused | any | **111** | 352 | 3.17 |
| `v2m2k_lean` | lean | any | 127 | 269 | 2.12 |
| `v2m2k_tight` | fused | any | **86** | 241 | 2.80 |
| `tcn3_lean` | lean | 3 | 113 | — | — |
| `tcn7_lean` | lean | 7 | 161 | — | — |
| `tcn15_lean` | lean | 15 | 209 | 488 | 2.33 |
| `tcn63_lean` | lean | 63 | 305 | 799 | 2.62 |
| `xfmr7_lean` | lean | 7 | 557 | — | — |
| `xfmr15_lean` | lean | 15 | 565 | 1,737 | 3.07 |
| `xfmr63_lean` | lean | 63 | 565 | 1,514 | 2.68 |
| `xfmr15_tight` | fused | 15 | 473 | 1,273 | 2.69 |
| `xfmr63_tight` | fused | 63 | 473 | 1,504 | 3.18 |

**The scaling law, one line per family:**

- **recurrence — O(1) in reach.** The state carries the past, so the step at reach 2 and the step at
  reach 630 are *the same step*: 111 ops fused, 161 lean, at every rung of §8af's ladder.
- **dilated convolution — O(log R).** Exactly **+48 ops per layer** (113 / 161 / 209 / 305 at
  L = 1 / 2 / 3 / 5) and `RF = 2^(L+1) − 1`, so cost ≈ 48·log₂(R+1) + c. Reaching 70 ms costs one
  extra layer over reaching 30 ms; reaching 630 ms costs two more.
- **banded attention — O(1) in reach, with a large constant.** The KV cache makes reach genuinely
  free (557 / 565 / 565 ops at bands 7 / 15 / 63; 473 / 473 fused), but the constant is
  **4.3× the fused recurrent step** and 5.5× the 2,362-parameter one.

**And the conversion is nearly a constant.** Excluding the stock `nn.LSTM` arm, min median ÷ op
count lands at **2.1–3.2 µs per operator** across a 6.6× range of operator counts, a **151× range
of parameters**, and all three families. That is §8ah's inferred law measured directly. The stock
arm is the sole outlier at 6.72 µs/op precisely because `nn.Module` call protocol is *not* an ATen
op and so never appears in the count — which is the whole of what the lean rewrite bought.

### 4. Verdict

1. **The cost model is `t ≈ N_ops × ~2–3 µs`.** Not parameters (151× buys 1.6×), not arithmetic
   (`v2m2k` does ≈ 2.4k MAC per step and spends microseconds on it). Operator dispatch is the unit,
   and this is now measured on both sides of the equation rather than inferred from one.
2. **Reach is free for recurrence, logarithmic for convolution, free-but-expensive for attention.**
   At reach 63: 111–161 ops recurrent, 305 convolutional, 473–565 attention. §8ah's ordering is
   confirmed by a machine-independent count, and so is its magnitude — 305/161 = **1.9×** and
   565/161 = **3.5×**, against the 1.3–1.8× and ~4.0× it measured in milliseconds.
3. **No absolute deadline is decidable from this pass — and that includes 10 ms.** Under the
   max-p99 rule **seven** arms fail even the 10 ms budget here (`gp_causal` 16.7, stock `seq_v2`
   15.1, `tcn15` 13.2, `tcn63` 40.8, `xfmr63` 32.4, `window_w2` 30.6, `v2m7k_lean` 14.1 ms) and
   three more land on the boundary. That is not a finding about those arms: on the minimum
   statistic the same seven sit at 2.5 / 1.7 / 3.8 / 5.6 / 2.2 / 4.4 / **0.6** ms — a
   6,866-parameter lean step "failing" 10 ms at 14.1 while its own best session is 0.64 is the
   contamination stated in one number — and **every lean and fused arm's min p99 is under 3 ms**.
   A pass that cannot reproduce a 10 ms result already established four times is measuring the
   room, not the model. So §8ah's "10 ms never binds" survives as a lower-bound statement, and
   1 ms is not resolved in either direction here.
4. **What the operator law implies for the next attempt.** Since cost is dispatch count, the lever
   with the most left in it is a runtime that collapses the ~111 remaining dispatches into a handful
   of kernels — a compiled/quantized export (ONNX Runtime, `torch.compile` with a static graph) —
   **not** further shrinking of the model, which §8ai has already taken to 1,808 parameters without
   the milliseconds following.

**What it does not show, and the measurement that would settle it** (§8j rule). Unchanged from
§8ah, plus one that this pass makes concrete: the 5 sessions must be run on a **quiet** machine
(nothing training, no network job), because that is the single difference between the 2.56× pass
and the 21.84× one. Until then the honest statement for the thesis is the operator count and the
ordering, not a millisecond.

### Addendum (2026-08-21) — the tcn2k step packed to 106 dispatches; jit.trace measured and rejected

**Motivation (승상님).** §8ai left tcn2k as the smallest arm of any family (1,808 params, ties the
backbone) but its online step was still the *shared* lean rewrite: 209 dispatched operators, slower
than the fused 357k backbone (111). The request was to give it the same treatment the LSTM got —
as few dispatches as the structure allows.

**`TightTCNStep`** (`b9_latency/tight_step.py`), three exact fusions mirroring `TightSeqV2Step`:
both branches' projection as one block matrix with structural zeros (routing enforced by layout,
§8ab), the three dilated-kernel taps as ONE `addmm` over a concatenated tap vector for both
branches at once (lean used six), and GELU/residual on the packed 16-wide vector. Only LayerNorm
stays per-branch (widths 10 and 6 normalize separately by definition: one `split`, two
`layer_norm`s, one `cat` per layer). The heads pack so the last matmul's two rows ARE
`[CES_TI, CES_VT]`. Equivalence vs the model's batch forward: max diff **2.1e-7** over a 40-step
replay (`make_arm`'s gate), and the full `op_count.py` table regenerated with every arm's gate
passing (46/46 pytest).

| step | aten ops / step | note |
|---|---:|---|
| `tcn2k_lean` (was the only option) | 209 | identical to `tcn15_lean` — count is structural, width-free |
| **`tcn2k_tight`** | **106** | below the fused backbone (111); 1.23× `v2m2k_tight` (86) |
| `_TCNStepPure` + `jit.trace`+`freeze` | 202 | **rejected** — measured once, not an arm |

**The jit result is the instructive negative.** Tracing requires tensor-state ring buffers, and the
shift (`cat` + `slice`/`narrow`/`select` per layer) costs more dispatches than the python-list
rotation it replaces, which costs zero. At this size the fuser recovered none of it: 202 > 106,
and 321 vs 279 µs on a busy machine. So "compile the step" (§8aj's own next-lever guess) is NOT
free at 1.8k parameters — the compiled route only pays if it collapses the whole step into genuinely
fused kernels (ONNX Runtime / a custom kernel), not via `torch.jit` on eager ops. The projected
time for 106 ops under the measured 2–3 µs/op law is **0.21–0.32 ms**; the milliseconds themselves
are deferred to a quiet-machine pass per this section's own rule (a GPU training job was running).

---

## 8ak. B.9 per-family reach ladder (2026-08-20) — the 70 ms threshold is **not** the LSTM's; it is the attention arm that moves

> **Extended and softened by §8al (2026-08-20).** `tcn5` and `xfmr5` fill the one rung in
> the 30–70 ms gap all three families can stand on. Attention is now measured *below* its
> crossing (50 ms, 2/4), so that crossing is bracketed rather than bounded — but it also **ties**
> the same-reach LSTM at 50 ms (−0.009), so "differs" fires at exactly one rung and the
> family effect is at most 0.023 anywhere.

**Question (승상님).** §8af measured the threshold on **one** family and every section since has
applied it to all three. §8ag then compared families at *matched* reach — but only at rungs 15 and
63, both far above the threshold. So the load-bearing rung had never been tested: **train each
family at 30 and 70 ms and see where its own ladder turns.**

**Design** (`experiments/b9_family/run_b9_family.py --arms tcn3 tcn7 xfmr7`, 12 runs, artifacts
`data/.b9_{tcn3,tcn7,xfmr7}_s*`, verdict `data/.b9_family.json`, figure + verdict block
`experiments/b9_family/reach_ladder_chart.py` → `docs/paper/figures/reach_ladder_by_family.*`).
Same `GATE_ENV`, same frozen manifests, same 4 split seeds. Each arm is paired against the axis A
LSTM rung at its **own** reach, so family is the single variable. `RF = 2^(L+1) − 1` puts the
convolutional rungs on 3 and 7 exactly; the attention band is set to 7 directly. There is **no
attention rung below 70 ms** — a band of 3 was not trained, which bounds one reading below.

### 1. The three ladders (`CES_TI` skill vs `gp_causal`, mean of 4 splits; `n`/4 = CI clears zero)

| context | recurrent (LSTM) | dilated convolution | banded attention |
|---:|---:|---:|---:|
| 20 ms | +0.055 · 2/4 | — | — |
| 30 ms | +0.083 · **3/4** | +0.079 · **3/4** | — |
| **70 ms** | +0.115 · **4/4** | +0.111 · **4/4** | +0.094 · 3/4 |
| 150 ms | +0.117 · 4/4 | +0.130 · 4/4 | +0.119 · **4/4** |
| 310 ms | +0.133 · 4/4 | — | — |
| 630 ms | +0.138 · 4/4 | +0.124 · 4/4 | +0.122 · 4/4 |

### 2. Paired against the LSTM rung at the **same** reach (§3.2 verdicts)

| arm | reach | params | `CES_TI` | w/l | verdict | `CES_VT` | w/l | verdict |
|---|---:|---:|---:|---|---|---:|---|---|
| `tcn3` | 3 | 71,442 | −0.004 | 0/0 | **tie** | +0.001 | 0/1 | tie |
| `tcn7` | 7 | 128,034 | −0.004 | 0/1 | **tie** | +0.003 | 0/0 | tie |
| `tcn15` | 15 | 184,626 | +0.014 | 2/0 | undecided | +0.000 | 0/0 | tie |
| `tcn63` | 63 | 297,810 | −0.016 | 0/2 | undecided | −0.036 | 0/1 | undecided |
| **`xfmr7`** | 7 | 295,746 | **−0.023** | **0/3** | **differs** | +0.007 | 0/0 | tie |
| `xfmr15` | 15 | 295,746 | +0.002 | 0/0 | tie | +0.022 | 0/0 | undecided |
| `xfmr63` | 63 | 295,746 | −0.019 | 0/1 | tie | −0.024 | 0/1 | undecided |

### 3. Verdict

1. **The threshold is a property of the problem for the two families that can be measured across
   it.** Recurrence and dilated convolution turn at the *same* rung and agree at every rung they
   share: 3/4 at 30 ms, 4/4 at 70 ms, and paired ties at both (−0.004, no significant split either
   way). A 71,442-parameter convolution reproduces a 357,570-parameter LSTM's threshold to within
   0.004 skill. **"70 ms" survives the family test it had never been given.**
2. **Attention is the one family that moves it, and it moves it *later*.** At 70 ms `xfmr7` scores
   +0.094 (3/4) and is **−0.023 below the LSTM at the same reach with 3/4 significant losses** —
   `differs` under §3.2, and the **only** such verdict anywhere in B.9's family axes. It reaches
   4/4 at 150 ms and ties the LSTM from there up. So the same architecture that is the most
   expensive per step (§8aj: 473 dispatched operators, 4.3× the recurrent step) is also the one
   that needs the most context to reach the same skill. Nothing recommends it on this problem.
3. **This bounds §8ag rather than overturning it.** §8ag concluded the families are
   indistinguishable "once they see the same 70 ms" — but its arms were trained at 150 ms and
   630 ms, so it never observed 70 ms. The corrected statement is **"once they see 150 ms or
   more"**; §8ag's wording is amended in place. Its magnitude claim is untouched and, if anything,
   sharpened: the largest family effect at matched reach anywhere in B.9 is **0.023**, while
   moving reach from 20 ms to 70 ms moves `T_i` by **+0.060** — still 2.6× larger, and the reach
   effect is 4/4 significant where the family effect appears at exactly one rung.
4. **Which bar you read changes the headline by one rung, so state the bar.** At the project's own
   promotion criterion (**≥ 3/4**) all three families clear at 70 ms and the threshold is
   unqualifiedly family-invariant. At the stricter **4/4** reading used for the figure's filled
   markers, attention needs 150 ms. Both are reported; neither is rounded toward the tidier story.
5. **`CES_VT` again settles nothing, and says so consistently.** No rung of any family reaches 3/4
   (best: `xfmr7` +0.162, 2/4). Every paired `V_rot` contrast at 30 and 70 ms is a tie. The target
   that rides carried input rather than fast-diagnostic history is indifferent to both reach and
   family — which is what §8ab's routing predicts and what §8af already found for the LSTM alone.

**What it does not show, and the measurement that would settle it** (§8j rule). The attention
ladder starts *at* the threshold: `xfmr3` was never trained, so "attention crosses ≥3/4 at 70 ms"
is an upper bound, not a turn. The named measurement is **a band-3 attention arm on the same four
splits** — 4 runs — which would say whether attention's ladder is the LSTM's shifted by one rung or
a different shape entirely. Second, all three convolutional rungs below 15 also change *width* with
depth (71k → 128k → 185k parameters, since layers carry channels), so `tcn3`/`tcn7` vary reach and
size together; the tie with the full-width LSTM at each rung is what makes that confound
non-fatal, but a width-held convolutional ladder would remove it outright.

---

## 8al. The dense ladder and a generality audit (2026-08-20) — saturation is **50 ms**, the 4/4 count cannot localize a threshold, and `V_rot`'s advantage is **not shot-general**

Three corrections, all from measurements this section adds rather than from re-reading old ones.
Two of them retire numbers this file has been quoting since §8af.

**Question (승상님).** "왜 2, 3, 7, 15만 실험하냐" — the ladder was spaced `2^(L+1) − 1` because that
is a dilated TCN's receptive field, and the LSTM was put on those rungs so "same reach, only the
operator differs" would be true. **The LSTM never needed that constraint**, and the widest unmeasured
gap (30 → 70 ms) is exactly where §8af placed its threshold.

**Design.** `run_b9_reach.py --reaches 4 5 6 10` (16 runs, `data/.b9_v2r{4,5,6,10}_s*`) fills the gap
one step at a time and checks that the plateau starts at 7 rather than somewhere before 15.
`run_b9_family.py --arms tcn5 xfmr5` (8 runs) adds the one rung in that gap the other two families
can stand on — at kernel 3 a TCN's `RF = 1 + 2·Σd` is **always odd**, and a 2-layer attention band
gives `RF = 2(band−1) + 1`, also odd, so 40 / 60 / 100 ms are unreachable for both without changing
the kernel or the depth. `tcn5` uses dilations (1, 1) and carries the *same* 128,034 parameters as
`tcn7`, so reach 5 vs 7 is size-controlled by construction.

### 1. Saturation is 50 ms, not 70 (`CES_TI`, paired vs the full-block backbone)

| context | 20 | 30 | **40** | **50** | **60** | 70 | **100** | 150 | 310 | 630 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| deficit | −0.066 | −0.033 | **−0.021** | **−0.017** | **−0.005** | +0.002 | **−0.009** | +0.005 | +0.023 | +0.029 |
| sig. w/l | 0/4 | 0/3 | 0/1 | 0/1 | 1/0 | 1/0 | 1/0 | 1/1 | 2/0 | 3/0 |
| §3.4 saturated | — | — | — | **yes** | yes | yes | yes | yes | yes | yes |

PREREGISTRATION_B9.md §3.4 is "the smallest r whose deficit is under `PRACTICAL_EPS` with ≤ 1/4
significant deficits". That rule now returns **5 steps = 50 ms**. §8af returned 7 only because 4, 5
and 6 did not exist; 7 was the first rung below the bar, not the smallest. Reach 4 misses by
**0.0009** — it is on the line, not clearly outside it.

**On this rule the ladder is a clean step**: not saturated at 20/30/40, saturated at every rung from
50 upward including 100. Quote **50 ms**.

### 2. The 4/4 count is not a function of reach at one-step spacing

Against `gp_causal`, the number of splits whose CI clears zero reads, along the recurrent ladder:

`2/4 → 3/4 → 4/4 → 3/4 → 4/4 → 4/4 → 3/4 → 4/4 → 4/4 → 4/4`  (20 … 630 ms)

It **falls back at 50 ms and again at 100 ms**, while the point estimates underneath rise smoothly
(+0.055 → 0.083 → 0.094 → 0.098 → 0.109 → 0.115). That is what a five-level vote on four samples
does; nothing is wrong with the runs. The coarse ladder hid it — with only {2, 3, 7, 15} measured,
"first 4/4 = 70 ms" looked like a clean step.

**Consequences, both applied here.** (a) The "first rung reaching 4/4" is retired as a threshold
locator; it was introduced by §8ak's figure and was never a pre-registered rule. (b) The figure and
the verdict block now use the project's **own promotion bar, ≥ 3/4**, on which every family's ladder
*is* monotone. Both counts stay in the table. No rung was dropped — deleting the 100 ms point was
considered and rejected, since 50 ms falls back too and the deletion would have bought nothing but
a tidier-looking curve.

### 3. The per-family ladders, read on the promotion bar

| context | recurrent | dilated conv | banded attention |
|---:|---|---|---|
| 20 ms | +0.055 · 2/4 | — | — |
| **30 ms** | **+0.083 · 3/4** | **+0.079 · 3/4** | — |
| 50 ms | +0.098 · 3/4 | +0.103 · 3/4 | +0.090 · 2/4 |
| **70 ms** | +0.115 · 4/4 | +0.111 · 4/4 | **+0.094 · 3/4** |
| 150 ms | +0.117 · 4/4 | +0.130 · 4/4 | +0.119 · 4/4 |
| 630 ms | +0.138 · 4/4 | +0.124 · 4/4 | +0.122 · 4/4 |

Two crossings are bracketed by measurement: recurrence is below the bar at 20 ms and above at 30;
attention is below at 50 and above at 70. The convolutional arm clears at **30 ms, its structural
minimum** (one layer, RF 3) — it cannot be asked to go lower, so its crossing is an upper bound.

Matched-reach pairing — the robust contrast — is quieter than the ladder: `tcn3` / `tcn5` / `tcn7`
tie the same-reach LSTM (−0.004, +0.005, −0.004; no significant split either way), `xfmr5` is
−0.009 (0/1, tie) and `xfmr7` is −0.023 (0/3, **differs**). So attention sits slightly below at
*both* low rungs in the same direction and the pre-registered rule fires at one of them.
**§8ak's "attention needs one more rung" is kept but softened**: the operator moves `T_i` by at most
0.023 anywhere, against +0.060 for moving reach from 20 to 70 ms.

### 4. `V_rot` is not underpowered — its advantage is not shot-general

The project has attributed `V_rot`'s 2/4 to the ~96-discharge power ceiling since §8p. That is
**wrong**, and the per-shot decomposition of the frozen B.1 artifacts says so directly.

| comparison | shots | fraction of shots the model wins | pooled |
|---|---:|---:|---:|
| `T_i` vs `gp_causal` | 96 | **0.695** (0.65–0.75) | +0.078 … +0.138 |
| `T_i` vs PCHIP | 96 | 0.557 | +0.174 … +0.264 |
| **`V_rot` vs `gp_causal`** | ~62 | **0.481** (0.40–0.53) | +0.020 … +0.331 |
| `V_rot` vs PCHIP | ~62 | 0.329 | +0.135 … +0.390 |

A per-shot win rate of **0.48 is a coin flip**, and the median shot's `V_rot` skill is ≈ 0
(−0.049 / −0.017 / +0.028 / +0.006). The positive pooled number comes from a handful of discharges:

| split | pooled | drop top-1 shot | drop top-3 | drop top-5 | top-1 share of the total gain |
|---|---:|---:|---:|---:|---:|
| 42 | +0.331 | +0.080 | +0.031 | **−0.006** | **84%** |
| 1 | +0.130 | +0.075 | +0.018 | **−0.041** | 49% |
| 7 | +0.020 | **−0.027** | −0.087 | −0.137 | **164%** |
| 123 | +0.134 | +0.048 | +0.015 | **−0.037** | 72% |

**Removing five of ~62 discharges takes every split to zero or below.** On split 7 a single shot
more than offsets the net loss on all the others. The `T_i` control behaves oppositely — top-1 share
**9–16%**, and dropping five shots leaves +0.045 … +0.113.

1. **More discharges will not fix this.** A shot-clustered bootstrap on a 0.48 win rate straddles
   zero at any n; B.6's 12 μs shots certainly cannot move it. The existing
   `hires_shots/power_analysis.py` already found that power here is set by *consistency*, not by
   effect size — this is that finding applied to the comparison that actually matters.
2. **The sentence changes.** Not "`V_rot` is underpowered at 96 discharges" but **"the `V_rot`
   advantage is not shot-general: the model wins on a minority of discharges by large margins and
   loses on the rest."** That is more specific, more defensible, and it names its own follow-up.
3. **The peak stratum does not explain which discharges.** Splitting shots by their peak fraction
   gives a win rate of 0.435 for the high-peak half against 0.528 for the low-peak half — the wrong
   direction. §8's "`V_rot` skill lives in the peak stratum" was a *row-level* result against PCHIP;
   it does not transfer to the shot level against the causal GP. **§8an found the variable that
   does work** — the discharge-level spread of the truth, which is not the same quantity as the
   row-level peak flag.

**What it does not show, and the measurement that would settle it** (§8j rule). Which discharges the
model wins on is now the open question, and peak fraction is ruled out. The named measurement is a
**per-shot covariate regression** on the frozen artifacts — win/loss against `gp_causal` versus
campaign position, mean gap length, held fraction, `T_e` level, and independent-observation count —
which costs no training and either finds the regime or establishes that the wins are idiosyncratic.
Until then `V_rot` is reported as a regime-conditional result, not a power-limited one.

---

## 8am. Pooled re-scoring of the reach ladder (2026-08-20) — the threshold buys **generality**, and a tight interval is not the same as a general effect

**Question (승상님).** "4/4 → 3/4 → 4/4로 튀면 문맥 증가 → 정확도 증가라는 결론에도 도달할 수 없는
상황이고, seed를 하나만 놓고 실험하기에도 애매하다." Both halves are right about the *statistic*.
Neither is right about the *data*: the point estimates underneath the flicker rise smoothly. The
fault is that a count of four binary outcomes discards the magnitude and precision each split
actually produced.

**Design** (`experiments/b9_reach/pooled_ladder.py`, no training — every number comes from the
frozen `comparison_errors_test.npz` of runs already on disk). Instead of four verdicts on 96
discharges, one verdict on the union: per-row squared errors from all four splits are concatenated
and bootstrapped with the **physical discharge** as the cluster, the same rule
`largegap/analyze_largegap.py` uses. Two facts license it, both measured first:

- **Overlap is mild.** Each split tests 96 of a 301-discharge union; **224 appear in exactly one
  split**, 72 in two, 5 in three or more; pairwise overlap is 10–20%.
- **Reuse is absorbed.** A discharge in two splits is *one* cluster carrying both its row sets, not
  two draws — conservative in the right direction.

And the ladder gets the test the count could never give: the **slope of skill against log context**,
refitted inside every bootstrap resample so its interval carries the same clustering.

### 1. `CES_TI`, pooled over 301 discharges (recurrent family)

| context | skill vs causal GP [95% CI] | win rate | −top10 |
|---:|---:|---:|---:|
| 20 ms | +0.057 [+0.027, +0.085] | **0.52** | +0.028 |
| 30 ms | +0.087 [+0.061, +0.111] | 0.60 | +0.060 |
| 50 ms | +0.104 [+0.079, +0.128] | 0.64 | +0.077 |
| 70 ms | +0.119 [+0.095, +0.142] | 0.66 | +0.092 |
| 150 ms | +0.123 [+0.096, +0.148] | 0.66 | +0.096 |
| 630 ms | +0.143 [+0.118, +0.168] | 0.67 | +0.116 |

"win rate" = the fraction of discharges on which the model beats the causal GP. "−top10" = the
pooled skill after deleting the ten best-contributing discharges. Both are printed beside the
interval on purpose; §3 says why.

**Trend: +0.0497 skill per decade of context [+0.0363, +0.0638] — rises.** One test, one interval,
no rung-by-rung flicker.

### 2. Two corrections this forces

1. **The model beats the causal GP at *every* measured context, 20 ms included.** The pooled CI at
   20 ms is [+0.027, +0.085]. The per-split rule read that rung as 2/4 = below the bar, and §9.1
   turned it into "70 ms (then 50 ms) is the context needed to beat the best available alternative".
   **That sentence is wrong.** The threshold was never a crossing point.
2. **What the threshold actually marks is saturation — and, better, generality.** Read the win-rate
   column: **0.52 → 0.60 → 0.64 → 0.66**, flat from ~70 ms. At 20 ms the model beats the causal GP
   on barely half the discharges while still winning on average; by 70 ms it wins on two thirds.
   **Context does not raise the average so much as it makes the win typical.** That is a better
   statement of the same finding and it is the one the deployment audience needs.

### 3. A tight interval is not a general effect — and `V_rot` is the proof

Pooling narrows every interval, and on `CES_VT` it produces a result that would be badly misread on
its own: **every rung's pooled CI clears zero** (e.g. 70 ms: +0.123 [+0.036, +0.248]). Read alone,
that reverses years of "`V_rot` is not resolved".

It should not be read alone. The same rows say:

| | discharges | pooled | win rate | −top5 | −top10 | top-1 share |
|---|---:|---:|---:|---:|---:|---:|
| `T_i` (v2r63) | 301 | +0.143 | 0.67 | +0.124 | +0.116 | 8% |
| **`V_rot` (v2r63)** | **197** | **+0.121** | **0.46** | **+0.021** | **−0.022** | **29%** |

**Deleting ten of 197 discharges flips the `V_rot` sign; deleting ten of 301 barely moves `T_i`.**
The bootstrap does not catch this, and the reason is mechanical rather than a flaw: with ~200
clusters a resample almost always contains several of the carrying discharges, so the interval is
genuinely reproducible *for this discharge set* while the effect is still not typical of a
discharge. The interval answers "would re-drawing these discharges agree"; the win rate answers "is
the win typical". **On `V_rot` those two questions have opposite answers, and the second one governs
the claim.**

The runner therefore prints both and flags the disagreement (`*!` = interval clears zero AND the
effect is not shot-general: win rate ≤ 0.60 or −top10 ≤ 0). Every `V_rot` rung of every family is
`*!`. `CES_TI` is `*!` only at 20 ms and, for the SSM, at 150 and 630 ms.

### 4. The other three families

| family | `T_i` trend per decade | `V_rot` trend |
|---|---|---|
| recurrent | **+0.050 [+0.036, +0.064]** rises | +0.003, not resolved |
| dilated conv | **+0.032 [+0.020, +0.044]** rises | **−0.016 [−0.043, −0.003] declines** |
| banded attention | **+0.031 [+0.011, +0.053]** rises | **−0.014 [−0.038, −0.003] declines** |
| diagonal SSM | +0.004 [−0.007, +0.014] **not resolved** | **−0.019 [−0.038, −0.009] declines** |

1. **Three families rise with context; the SSM does not.** Its `T_i` skill is flat at ~+0.10 from
   70 ms to 630 ms — it does not convert extra context into skill, which is a sharper statement of
   its paired result (it loses to the same-reach LSTM by −0.022 at 7 and −0.044 at 63) and an
   independent reason not to adopt it here.
   **[Corrected 2026-08-21 — addendum below. With rungs 2/3/5 bought, the six-rung trend RISES
   (+0.0235 [+0.0102, +0.0372]): the 70–630 ms flatness was the saturated tail of a curve whose
   rising region had simply never been measured.]**
2. **`V_rot` gets *worse* with context for three of four families**, significantly. More context is
   not neutral for the target that rides carried values — it is harmful. This is consistent with
   §8ab's routing story (the `V_rot` branch reads a slowly-varying carried value; a longer window
   adds variance, not signal) and it is a new, testable statement rather than a null.

### 5. Verdict

1. **The count is retired as an estimator, and pooling replaces it.** One interval over 301
   discharges, plus a trend test, in place of four verdicts over 96. No run was discarded and no
   rung was dropped.
2. **"More context, more accuracy" is established for `T_i`** by a single pre-specified trend test
   (+0.050 per decade, CI clears zero) in three of four families, rather than inferred from a
   flickering vote.
3. **"You need N ms to beat the causal GP" is withdrawn.** The model beats it at 20 ms already. What
   N marks is where the win becomes *typical* (win rate 0.52 → 0.66) and where the deficit against
   unbounded context falls under the practical floor (§8al: 50 ms).
4. **Pooling does not rescue `V_rot`, and the audit columns are why we know.** A narrower interval
   made an effect carried by 5% of discharges look settled. Any future pooled result in this project
   is reported with the win rate and −top10 beside it.

**What it does not show, and the measurement that would settle it** (§8j rule). Pooling four splits
estimates the *procedure's* expected skill, not one model's — the four runs per rung are four models
trained on different data, and their errors are concatenated. That is the right estimand for "does
this method work", and the wrong one for "will this checkpoint work"; a deployment claim still needs
a single trained model on a held-out campaign (§8's temporal split). And the mild split overlap
(72 discharges in two test sets) is absorbed by the clustering but not removed: a fully clean design
is **discharge-level k-fold**, where every discharge is tested exactly once, which would cost the
same number of runs and remove the caveat entirely.

### Addendum (2026-08-21) — the SSM's missing low rungs bought; "does not convert context" is corrected

**Motivation (승상님: "지금 ssm 사다리 왜 안돌려").** The not-resolved SSM trend above was
estimated on three rungs, 70–630 ms — every one of them PAST the region where the other three
families turn. A trend test with no leverage where the action is cannot resolve anything, and this
project's own rule (§8j) says the negative must name its measurement. The measurement was rungs
2/3/5: 12 runs, `run_b9_reach.py --variant ssm`, the same frozen protocol as every other rung.

**Pooled (301 discharges), the full six-rung SSM ladder, `CES_TI` vs the causal GP:**

| context | skill [95% CI] | win rate | −top10 |
|---:|---:|---:|---:|
| 20 ms | +0.065 [+0.036, +0.092] | 0.49 | +0.033 |
| 30 ms | +0.081 [+0.052, +0.106] | 0.57 | +0.052 |
| 50 ms | +0.098 [+0.074, +0.121] | 0.58 | +0.071 |
| 70 ms | +0.100 [+0.077, +0.124] | 0.60 | +0.074 |
| 150 ms | +0.106 [+0.080, +0.133] | 0.58 | +0.077 |
| 630 ms | +0.105 [+0.080, +0.131] | 0.58 | +0.076 |

1. **The trend verdict flips: rises.** +0.004 [−0.007, +0.014] over three rungs becomes
   **+0.0235 [+0.0102, +0.0372] per decade over six**. The sentence "it does not convert extra
   context into skill" is withdrawn. The correct sentence: **the diagonal SSM converts context up
   to ~70 ms and then stops, at a lower ceiling** — +0.105 against the recurrent +0.143 at 630 ms,
   win rate capped at 0.58–0.60 against 0.66. Its deficiency is long-context exploitation, not
   context conversion.
2. **At 20 ms it is the best pooled point of any family** (+0.065 vs the recurrent +0.057) —
   a same-rung pooled comparison, not a paired test, so it is reported and not promoted. If the
   minimal-context regime ever matters (a 20 ms deployment), the named measurement is a paired
   `ssmr2` vs `v2r2` comparison, which the existing artifacts already support.
3. **The adoption verdict does not change.** The SSM still loses paired to the same-reach LSTM
   at 7 (−0.022) and 63 (−0.044) and tops out lower. What changed is the *reason* on record —
   and that the §3.4-style saturation read for this family lands at ~70 ms like the others.
4. **`CES_VT` declines with context in this family too** (−0.0134 [−0.0319, −0.0037]), closing
   statement 2 above from "three of four families" to **four of four**.

Artifacts: `data/.b9_ssmr{2,3,5}_s*`, `data/.b9_reach_ladder_ssm.json`,
`data/.b9_pooled_ladder.json` regenerated (the other three families' rows reproduce unchanged —
their bootstrap draws precede the SSM's in seed order); the context-family figure regenerated.

---

## 8an. Which discharges the model wins on (2026-08-21) — it wins where the quantity moves, and that closes `V_rot`'s open question

**Question.** §8am replaced "`V_rot` is underpowered" with "`V_rot`'s advantage is not
shot-general", which is more accurate and still only a description. §8j forbids stopping there:
a negative result has to name the measurement that explains it. The named measurement was a
per-discharge covariate regression, and this is it.

**Design** (`experiments/b9_reach/shot_covariates.py`, no training — every column comes from the
frozen `comparison_errors_test.npz` of the `v2r63` runs, pooled over the four splits). One row per
physical discharge: its skill against the causal GP, plus eight covariates the artifacts already
carry — campaign position (the shot number), scored rows, mean and max gap, peak fraction, the
truth's level and spread within that discharge, and the causal GP's own RMSE there. `CES_TI` is
carried as the control, since its win *is* shot-general (69.9%). Significance is by permutation
(10⁴ shuffles) and Bonferroni-adjusted across the eight; **exploratory — no rule was
pre-registered, so nothing here promotes or demotes a claim.**

### 1. Only two covariates survive, and they are the same covariate

| covariate | `CES_TI` ρ | p (Bonf.) | `CES_VT` ρ | p (Bonf.) |
|---|---:|---:|---:|---:|
| campaign position | +0.044 | 1.00 | +0.042 | 1.00 |
| scored rows | +0.015 | 1.00 | +0.083 | 1.00 |
| mean gap | +0.140 | 0.15 | +0.081 | 1.00 |
| max gap | +0.086 | 1.00 | +0.064 | 1.00 |
| **peak fraction** | +0.116 | 0.42 | −0.005 | 1.00 |
| target level | +0.305 | **0.0008** | +0.055 | 1.00 |
| **target spread** | **+0.401** | **0.0008** | **+0.281** | **0.0024** |
| **causal-GP RMSE there** | **+0.373** | **0.0008** | **+0.318** | **0.0008** |

The last two are not separable: `target spread` and `baseline RMSE` correlate at **ρ = 0.85** on
both targets, because a discharge whose truth moves is exactly a discharge the GP interpolates
badly. Partialling one out leaves the other weak (`T_i`: +0.170 / +0.067; `V_rot`: +0.028 / +0.156),
so the honest statement is one covariate — **how much the quantity moves in that discharge** — not
two.

**And `baseline RMSE` alone would have been an artifact trap**: it is the denominator of the skill
ratio, so a discharge where the GP does badly scores high skill more easily. The check that removes
the ratio entirely is the **binary win/loss**, and it holds:

| | ρ(spread, win) | p (perm.) |
|---|---:|---:|
| `CES_TI` | **+0.393** | 0.0001 |
| `CES_VT` | **+0.198** | 0.0075 |

### 2. Read as terciles of how much the target moves

| tercile of target spread | `CES_TI` win rate | median skill | `CES_VT` win rate | median skill |
|---|---:|---:|---:|---:|
| quiet | **42%** | −0.051 | **34%** | −0.091 |
| middle | 83% | — | 48% | — |
| variable | **85%** | +0.156 | **55%** | +0.031 |

### 3. Verdict

1. **The mechanism is physical, not statistical.** The model wins where the quantity *moves*.
   Where a discharge is flat, the causal GP is already near-optimal — interpolating a straight line
   is easy — and the model loses slightly by adding variance to a problem that had none. The 46%
   pooled `V_rot` win rate is not mysterious: **most discharges are quiet, so most discharges have
   nothing to win.**
2. **`peak fraction` is not the variable, and that corrects §8am.** §8am split discharges by peak
   fraction and got the wrong sign (0.435 high-peak vs 0.528 low-peak). The peak flag is a
   *row-level* activity proxy built from neighbour slopes; what predicts the win is the
   *discharge-level spread of the truth*. Different quantity, and only the second one works.
3. **This is why `T_i` is shot-general and `V_rot` is not — and it is only half the reason.** In
   the most variable third `T_i` wins **85%** and `V_rot` only **55%**. Variability explains why
   quiet discharges are lost; it does **not** explain why `V_rot` barely wins even when it moves.
   That residual points straight at §2.3d: **the driver of rotation is not in the dataset.** A model
   cannot track motion it has no input for, so it wins the variable discharges only slightly, by
   carrying the last observation better than a GP does. Two independent lines of evidence — the
   `T_e`~`V_rot` null (r = +0.024, p = 0.58) and this tercile table — now say the same thing.
4. **`V_rot`'s open question is closed as far as this dataset can close it.** The sentence is:
   *the advantage concentrates in discharges where rotation actually varies, and even there it is
   small, because the quantity that drives the variation is unobserved.* That is a mechanism, a
   measurement, and a named data lever — not "insufficient information".

### 4. Addendum (same day) — the three §8al-named covariates the artifacts do not carry

§8al named five covariates; the first pass tested what the npz files hold and left out the held
fraction, the independent-observation count and the `T_e` level. This pass completes the list —
the first two from the hires_shots census (`shot_metrics.csv`, computed under the confirmed
protocol's treatment), the `T_e` level as the discharge's mean raw ECEI (the §8b.3 proxy,
uncalibrated) — on its own rng stream, so §1's numbers reproduce **bit-for-bit** (verified against
the pre-addendum JSON); Bonferroni is over all 11 covariates tested per target. Merge is 286/286
and 174/174; the npz `shot` key is an index into `dataset.valid_files`, verified equal to
`sorted(data/s*.csv)` (all 641, in order, in every disk cache) before mapping to shot numbers.

| covariate | `CES_TI` ρ | p (Bonf.) | `CES_VT` ρ | p (Bonf.) |
|---|---:|---:|---:|---:|
| held fraction | +0.043 | 1.00 | −0.084 | 1.00 |
| independent observations | −0.006 | 1.00 | +0.114 | 1.00 |
| `T_e` level (mean ECEI) | +0.152 | 0.11 | −0.031 | 1.00 |

**None survives, and the verdict does not move.** Three readings: (a) all five covariates §8al
named are now measured, and the only variable that predicts the win remains how much the target
moves in that discharge; (b) the `V_rot`~`T_e`-level null is a **third** independent line for the
torque-unobserved mechanism, after §8b.3's row-level r = +0.024 and §3's tercile residual — the
discharges where the model wins on rotation are not the hot ones, they are the moving ones;
(c) `T_i`'s +0.152 raw is directionally consistent with `target_level` (+0.305) but does not
survive correction — report neither as a finding.

**What it does not show, and the measurement that would settle it** (§8j rule). This is
exploratory: eleven covariates, one arm, no pre-registered rule, and the two surviving covariates
are collinear. It also cannot separate "the model is good at variable discharges" from "the GP is
bad at them" beyond the binary check. The measurement that would settle it is **B.6**: if the μs
re-acquisition delivers a mode-rotation frequency, the prediction this section makes is specific
and falsifiable — the `V_rot` win rate should rise **in the variable tercile first**, because that
is where a rotation-carrying input has something to explain. If it rises uniformly, or in the quiet
tercile, this mechanism is wrong.

---

## 8ao. The μs shot set is frozen (2026-08-21) — the scan finishes by discovering a fourth way to be wrong, and test grows to four

**Question.** Which twelve discharges get the microsecond re-acquisition? The 2026-08-20
handoff left three opens: 253 unscanned shots, a provisional list marked for replacement,
and the test-3-vs-4 decision.

**Design.** (1) Finish the batched OpenAlex full-text sweep (`--priority`, V_rot-richest
first; positive/negative controls had already licensed reading an empty batch as absence).
(2) Hand-verify every new hit against a readable version (arXiv / Springer / Nature) before
believing it. (3) Re-rank with `score_v2`, assign roles literature-first
(`select_literature_first.py`), and put the one genuine open decision to 승상님.

**Results.**

1. **Scan complete, 641/641, in one budget day.** Fifteen new index hits — and every one
   rejected by hand-verification, which discovered the **fourth false-positive class: AIP
   article numbers.** PoP ids are `iissnn` (032302, 032111, 032309 …) and the index matches
   the bare five digits inside them, so any KSTAR paper citing a PoP issue-3 article "hits"
   the matching campaign shot number. Six-for-six direct proof: the readable versions carry
   the zero-padded id in their bibliographies and never the bare number. The
   `FALSE_POSITIVES` table now rejects 32004/32111/32115/32151/32301–32310 (evidence
   per shot); the positive control could not have caught this class because its decoys
   were out-of-range numbers that collide with nothing.
2. **One hit survived: #32092** (NF 2026 `ae8679`, KSTAR edge-harmonic-oscillation paper;
   `032092` is structurally impossible as an AIP id; IOP wall keeps the sentence unread —
   kstar-grade, same standard as #31747/#31097). The usable literature ledger is **11**.
3. **Decisions (승상님).** ① **test = 4**: literature-first keeps #31873 (V_rot fully held)
   in test, so k = 3 gives two effective `CES_VT` clusters — the measured false-positive
   regime (0.665 → 0.770 pass rate going k = 3 → 2). Adding **#31902** (412 V_rot, most of
   any gate-passing s42-test candidate) restores three effective clusters; measured power
   0.750 (`CES_VT`) / 0.368 (`CES_TI`). Price: pool 7 → 6 (#31914, 542 V_rot, lost its
   slot). ② **#32092 included** (pool) — rule consistency; it is also a top-tier clean
   Mirnov shot (RMS 20.5, trim 0.94, kurt 1.8, coherence 0.92).
4. **The frozen twelve** (roles in `folds.py`, windows in `fetch_windows.csv`, table in
   `hires_shots/SELECTION.md`): test 31921 · 31873 · 31114 · **31902**; pool 31097 · 31359
   · 31747 · 32027 · **32092** · 32097; companions 31923 · 31357. Request volume 65.66 s.
   Six of ten roles are literature shots. `PREREGISTRATION_B6.md` §1.2 (≥ 3 effective
   V_rot test clusters) **passes**: 296 / 311 / 412.

**Verdict.** The shot set is frozen and the B.6 execution gates §1.1–§1.3 are discharged;
only the ⟦on-arrival⟧ slots (sampling rates) remain. The scan's lasting lesson joins the
false-positive taxonomy: **a number that survives the machine filters is still not a
discharge until a readable version shows it in a sentence** — the AIP-id class produced
eleven plausible-looking literature shots in one day, several with excellent V_rot
statistics, and accepting them would have handed role slots to bibliography artifacts.

**What it does not show, and the measurement that would settle it** (§8j rule). #32092's
citation is attributable but unread; the named measurement is an SNU institutional login to
IOP (`10.1088/1741-4326/ae8679`) — if the sentence names the discharge, upgrade to
confirmed; if the number is something else, #32092 drops to a data-only pool shot and the
slot goes to the next score_v2 candidate. And the k = 4 measured power is shot-cluster
only; the block-bootstrap sweep has no k = 4 row (recorded as absent in `folds.py`, worth
adding when `power_analysis.py` next runs).

### Addendum (same day) — the named measurement ran, and #32092 is CONFIRMED

승상님 fetched the full PDF through the IOP wall the same evening. The bare-digit grep
still found nothing — because **IOP typesets the discharge number with a thin thousands
space, "#32 092"**, which is also why the OpenAlex index (which normalizes it) hit while
every readable-text check missed. Six mentions, all substantive: it is **the paper's
representative QH-mode discharge** (fig. 2 time evolution, fig. 3 kinetic profiles,
fig. 4 peeling–ballooning at t = 4.9 s ELMy vs t = 7.6 s ELM-free, and table 1: **EHO
n = 1 with harmonics at ~4 kHz / ~8 kHz, measured on Mirnov coils + ECEI**). Ledger
verdict upgraded kstar → confirmed; the list does not change — the slot it holds is now
better justified, and the EHO frequencies become B.6 positive control #5
(`PREREGISTRATION_B6.md` §1.4): the one discharge with a *published answer inside the
3–10 kHz band H1 is trying to restore*. Two false-positive lessons sharpen: the AIP-id
class stands (those numbers were absent from every version), but "absent from the
extracted text" alone is not proof of absence — typographic variants (thin spaces) must
be grepped before rejecting a hit on a paper that names KSTAR. A same-day UFCES sweep of
all six readable papers (and OpenAlex full-text for the FIRE and edge-kink papers) found
**no ultra-fast CES anywhere in the twelve's literature**; the EPJ NTM paper stays
unchecked (unreadable + OpenAlex budget exhausted).

---

## 8ap. The quantum arm is closed (2026-07-26 → 08-24) — full record lives outside this file

**Verdict: closed, negative, hardware-verified. Nothing here feeds the main pipeline.**

A variational quantum circuit was compared against a parameter-matched classical MLP on
PCA-reduced CES inputs, then evaluated on real IonQ Forte hardware (55 circuits, $1,418.44 of
a K-Quantum Hub trial credit). Four independent axes are closed:

| axis | result |
|---|---|
| accuracy | the circuit loses to persistence on a **noiseless** simulator (471.5 vs 449.6 eV) — before hardware is involved at all |
| latency | 22.9 s per prediction against a 10 ms budget; ~1,800× over, and a 1000× faster gate set still overruns |
| hardware damage | depolarizing coefficient **λ = 0.661 ± 0.040** (16.7σ): Forte returns 34% of the intended signal amplitude |
| scaling | more input dimensions and more parameters both make the *classical* control worse, so there is no headroom a bigger circuit could exploit |

Three of the four QML families are closed (variational circuits, fixed feature maps, reservoirs);
quantum kernels tie their classical control (+0.3279 vs RBF +0.3307) but need O(N²) circuits on
hardware. A trained-quantum-encoder arm is **inconclusive** and labelled as such.

**Why this does not contaminate the confirmed results.** The checkpoint is the July
`quantum_vqc_weights.pt` at `window_size = 4`, the pre-reset protocol. No number from this arm is
quoted alongside the backbone's skill figures, and no quantum code is imported anywhere in
`ces_prediction/` outside `experiments/quantum/`.

**Full record** — design, billing model, the shot ladder, the λ measurement, the free family
sweep, two corrected misreadings, and the method lessons — is in
`docs/ionq_qpu_실험기록.md` (§8–17 and Appendix A). Code is `experiments/quantum/`.

## 8aq. The target's own reproducibility (2026-09-03) — the ceiling claim gets an absolute anchor, and it is **tail-shaped**

**Question.** Every ceiling statement so far is *relative*: a 21k-parameter rung matches the
backbone (§8z), a 26× width sweep is flat (§8aa), three operator families tie within 0.023
(§8ag). Together they say *a bigger estimator does not help*. None of them says **how much of
the residual is even reducible**. The scored target is a photon-integrated spectral fit, so it
carries its own measurement noise, and a perfect predictor of the true physical `T_i` would still
score `RMSE² = σ_meas² + (physics error)²` against it. This section measures σ_meas from the
observed series itself.

**Design** (`ces_prediction/analyze_noise_floor.py` → `data/.noise_floor.json`, full 641-file
census, no model involved). Difference-based estimators that are unbiased for white noise on a
smooth signal and shed signal bias as the order rises — order 1 (Rice) `σ² = E[d²]/2`, order 2
(GSJS) `/6`, order 3 `/20`, order 4 `/70` — plus a robust MAD variant of order 1, order-4
estimates trimmed at the 95th/99th percentile, and the semivariogram. Data treatment is the
confirmed protocol's: spikes cut **before** held detection, held repeats removed, and a difference
is formed only across consecutive same-block grid steps. **Every estimate is an upper bound**: any
real physics faster than the difference operator can annihilate is counted as noise, which is the
correct bound for a 10 ms nowcaster but is *not* a statement about the diagnostic's calibration.

### 1. What the target's own scatter is

| population | target | order 1 | order 2 | order 3 | order 4 | trimmed 95% | MAD (bulk) |
|---|---|---:|---:|---:|---:|---:|---:|
| cut | `CES_TI` (eV) | 151.8 | 139.9 | 133.8 | **129.9** | 68.7 | **46.4** |
| cut | `CES_VT` (km/s) | 28.8 | 22.8 | 19.5 | **16.4** | 6.8 | **4.3** |
| inclusive | `CES_TI` (eV) | 383.2 | 330.0 | 294.7 | **274.3** | — | 46.9 |

Runs of consecutive observed samples: 5,987 runs / 222,882 rows (`T_i`, cut), 2,800 / 84,833
(`V_rot`); longest run 463 / 149 steps. The order-1→order-4 sequence is **still falling at order 4**,
so 129.9 eV has not converged and remains an upper bound.

### 2. The estimate is tail-dominated — which is the actual finding

| population | target | top 1% of 4th differences | top 5% |
|---|---|---:|---:|
| cut | `CES_TI` | **46.6%** of the squared mass | 73.4% |
| cut | `CES_VT` | **65.6%** | 83.5% |

So the population has two regimes, and they give opposite answers:

- **In the bulk**, `CES_TI` is reproducible to **46–69 eV** against a backbone RMSE of 157.8 eV
  — the model is 2.3–3.4× above the target's own scatter, i.e. only 7–9% of its MSE is
  irreducible. **There is real headroom in the bulk.**
- **In the tail**, the order-4 bound of 129.9 eV is 68% of the backbone's MSE. The remaining
  error is concentrated where the target itself stops being reproducible.

This does not contradict §8z/§8aa/§8ag — it **explains** them. Width and operator family are flat
because the loss is a tail statistic: nearly half the squared mass sits in 1% of the one-step
changes, and no amount of capacity fits a target whose tail is fit failure. It is the same fact
§8ab found from the other side (≈ 1% of spike-anchor rows carrying 70–83% of every arm's squared
error), now measured on the whole population rather than on the `V_rot` anchors.

### 3. The semivariogram says the 10 ms grid is nearly pure nugget

`γ(1) = 23{,}051` for cut `CES_TI` — **15.9%** of the population variance in a single 10 ms step
— and `γ(1) > γ(2)` (23,051 vs 21,201), which is the signature of a dominant nugget rather than a
smooth process sampled finely. Adjacent CES samples do not lie on a smooth curve at 10 ms. Two
consequences: the power-law nugget fit (σ = 147.7 eV) is **not trustworthy** on a non-monotonic
short-lag variogram and should not be quoted; and `√γ(1) = 151.8 eV` is within 4% of the
backbone's 157.8 eV RMSE — the nowcaster errs by about what the diagnostic moves between
consecutive samples.

### 4. Verdict

1. **The absolute-scale sentence now exists.** The nowcaster reconstructs `T_i` to 157.8 eV against
   a target whose own 10 ms reproducibility is **46–130 eV** depending on how the tail is treated.
   Same order of magnitude — which is the physics-facing statement a skill score cannot make. Quote
   the range and the reason it is a range; never quote a single σ.
2. **The ceiling claim is corrected, not overturned.** "The ceiling is information, not the
   estimator" holds for the *aggregate* loss because the aggregate loss is a tail statistic. It does
   **not** hold in the bulk, where the model is still 2–3× above the target's scatter. Write it that
   way; the unqualified version overclaims.
3. **The named next measurement** (§8j rule — a negative result must name the fix): re-score the
   frozen artifacts under a **robust co-metric** (median absolute error, or a trimmed / Huber skill)
   alongside MSE-skill. It needs no retraining, and it answers directly whether +0.236 is a bulk
   effect or a tail effect. **It touches TEST, so it is pre-registered before it is run**, not after
   — the rule and the promotion bar go in the pre-registration first.

**What this section does not show.** No separation of measurement noise from genuine sub-10 ms
physics — the estimators bound their sum, which is the right bound for this task and the wrong one
for a statement about the instrument. No per-shot or per-regime breakdown. No model was scored,
selected, or re-selected here.

Artifacts: `data/.noise_floor.json`; script `ces_prediction/analyze_noise_floor.py`
(the LADDER_RMSE constants it prints ratios against are quoted from `docs/paper/main_ko.tex`
table `tab:ladder`, itself generated from the frozen artifacts).

---

## 8ar. Physical time scales, and a term-by-term audit of the two transport equations (2026-09-03) — the torque balance is now closed by measurement, not by assertion

**Question (승상님, 2026-09-03).** The thesis record is 47 sections of validation and the paper
carries nine tables, seven figures and **zero displayed equations**; its model section is 53 lines
against 596 for results. Every time scale in it is empirical and unanchored — the ~50 ms context
saturation (§8al), the 10 ms grid, the aliased Mirnov stream (§8b.2). Two things were asked: put a
physical scale beside each of those numbers, and **check whether the angular-momentum balance can
still reach `V_rot` through some term we have not tried.**

**Design** (`ces_prediction/analyze_physics_scales.py` → `data/.physics_scales.json`, full 641-file
census, confirmed data treatment: spikes cut before held detection, held removed, contiguous blocks
only). No model is trained, scored or selected; this is a data-and-theory audit in the class of
`analyze_data_evidence.py` and `analyze_noise_floor.py`.

### 1. The one term of the torque balance we had not tested — and it is null

The conservation laws constrain a **rate**, not a level:

- `d/dt (3/2 n_i T_i) = Q_ei (T_e − T_i) − ∇·q_i + …`
- `d/dt (n_i m_i ⟨R²⟩ ω_φ) = −∇·Π_φ + T_NBI + T_NTV + T_intrinsic + …`

So the honest question is not "do the fast diagnostics correlate with `V_rot`" — §8ab answered that
with an ablation — but **"do they correlate with `dV_rot/dt`"**. Reading the torque balance term by
term against what we hold:

| term | in our data? | status |
|---|---|---|
| LHS `dL/dt` | yes, finite difference on consecutive observed pairs | measured here |
| `T_NBI` | **no 0-D channel exists in the dataset** | absent (§8b.3) |
| `T_NTV ~ δB²` | Mirnov, but decimated to 100 Hz | **closed negative** (§8b.2): `\|MC\|` and rolling RMS already tried, no help |
| `∇·Π_turb` | BES sees the density fluctuation the turbulent flux rides on | **was never tested — tested here** |
| `T_intrinsic` | needs `∇T_i`; we hold a scalar | not reachable |

Pooled Pearson `r` over 624 blocks, fast-channel modality mean against the target, at lag 0
(`level` = the target itself, `rate` = its one-step increment):

| driver → target | level | **rate** |
|---|---:|---:|
| BES → `T_i` | **+0.341** | **+0.070** |
| ECEI → `T_i` | **+0.311** | **+0.078** |
| BES → `V_rot` | +0.027 | **−0.006** |
| ECEI → `V_rot` | +0.005 | **−0.003** |
| MC → `T_i` | −0.001 | +0.001 |
| MC → `V_rot` | −0.001 | +0.007 |

**The turbulent-flux term is null**, and the same measurement finds a signal for the energy
equation (`+0.070 / +0.078`), so the method can detect a rate coupling where one exists. With
`T_NBI` absent, `T_NTV` closed and `T_intrinsic` unreachable, **every term of the angular-momentum
balance that our inputs could carry is now accounted for by measurement.** "The driver of rotation
is unobserved" (§9.4) stops being a data-level remark and becomes a term-by-term audit of a
conservation law. This is also the correct place to state that the fix is acquisition, not
modelling — the paper's §8j rule.

### 2. Relaxation scales — clean for `T_i`, and **unmeasurable** for `V_rot`

Pooled autocorrelation, e-folding time (`1/e`):

| channel | r(10 ms) | r(50 ms) | r(100 ms) | r(300 ms) | 1/e |
|---|---:|---:|---:|---:|---:|
| `CES_TI` | 0.652 | 0.529 | 0.436 | 0.226 | **159 ms** |
| BES | 0.675 | 0.517 | 0.417 | 0.296 | 161 ms |
| ECEI | 0.660 | 0.525 | 0.432 | 0.261 | 147 ms |
| `CES_VT` (held-free) | 0.423 | 0.249 | 0.158 | 0.060 | 16 ms |
| `CES_VT` (held-kept) | 0.803 | 0.720 | 0.654 | 0.476 | **> 300 ms** |
| MC | −0.007 | −0.000 | −0.005 | −0.002 | (white noise) |

`T_i`, BES and ECEI share a relaxation scale to within 10% (147–161 ms) — `T_i` is clean here
because it carries **1 held row in 641 files**. `V_rot` is the opposite: 54% of its observations are
instrument repeats, so held-free and held-kept bracket its scale between **16 ms and > 300 ms**, a
factor of ≥ 19. **The `V_rot` relaxation scale cannot be measured from this dataset at all.** That is
a finding, not a gap in the analysis: the held pathology destroys not only 54% of the values but also
our ability to characterize the quantity's own dynamics — one more entry for the acquisition ledger.

### 3. A physical scale beside the ~50 ms context, for the first time

Electron-ion energy equipartition, `τ_eq = (m_i / 2 m_e) · τ_e` with the Braginskii/NRL electron
collision time `τ_e = 3.44×10⁵ · T_e[eV]^1.5 / (Z · n_e[cm⁻³] · lnΛ)`, deuterium, lnΛ = 17, Z = 1:

| `T_e` \ `n_e` | 2×10¹⁹ m⁻³ | 3×10¹⁹ m⁻³ | 5×10¹⁹ m⁻³ |
|---|---:|---:|---:|
| 500 eV | 20.8 ms | 13.8 ms | 8.3 ms |
| 1000 eV | 58.7 ms | 39.2 ms | 23.5 ms |
| 2000 eV | 166.1 ms | 110.8 ms | 66.5 ms |
| 3000 eV | 305.2 ms | 203.5 ms | 122.1 ms |

Our observed `CES_TI` median is 593 eV (p99 2,089 eV), so the CES-relevant corner of this table is
**8–59 ms**, which brackets the measured saturation of ≈ 50 ms (§8al). **This is the first time an
empirical threshold in this work has a physical scale beside it.** State it as an order-of-magnitude
consistency and nothing more (below).

### 4. The Mirnov loss as a derivation rather than a statistic

The grid is 10 ms, so the Nyquist frequency is **50 Hz**. Mirnov mode rotation is kHz — the shot
papers put KSTAR EHO harmonics at ~4 and ~8 kHz (§8ao). A 100 Hz decimation without an
anti-aliasing filter therefore *cannot* represent the mode, by the sampling theorem. §8b.2's
lag-1 autocorrelation of −0.009 (against BES +0.568) is the **consequence**, and the paper should
lead with the derivation and cite the statistic as its confirmation, not the other way round.

### 5. What this does not show

- **The autocorrelations are not detrended.** Blocks are ≈ 3 s (median 301 rows), so these numbers
  measure "how long a signal stays similar", dominated by the discharge envelope — they are **not**
  transport relaxation times, and must not be quoted as `τ_E` or `τ_φ`.
- **The lag profiles cannot resolve `τ_eq`.** Every `T_i` peak is at lag 0, not at the ~4 grid steps
  a 39 ms coupling delay would produce. The shared slow envelope dominates; the informative part is
  the *amplitude* (`r ≈ 0.3`), not the lag.
- **`τ_eq` uses assumed plasma parameters.** `n_e` and `T_e` in physical units are not in our CSVs
  (BES and ECEI are instrument units), so the table is quoted, not derived from our data, and the
  order-unity convention in `τ_eq` is stated rather than hidden.
- **The held-free `V_rot` autocorrelation is computed on a shifting subpopulation** (a lag-1 pair
  only exists where the instrument updated within 10 ms), which is a second reason not to quote
  16 ms as a physical time.
- No model was scored. The `r` values are data-level associations, not skill.

### 6. Verdict

1. **The torque balance is closed by measurement.** Four terms, four dispositions: LHS measured,
   `T_NBI` absent, `T_NTV` closed negative, `∇·Π_turb` null here, `T_intrinsic` unreachable. Write
   §9.4 that way — it converts the paper's weakest-looking result into its most rigorous negative.
2. **`T_i`'s predictability has a matching physical scale** (`τ_eq` 8–59 ms vs saturation ≈ 50 ms;
   `T_i`/BES/ECEI relaxation 147–161 ms), reported as order-of-magnitude consistency.
3. **`V_rot`'s own dynamics are unmeasurable in this dataset**, which is a third named entry for the
   acquisition ledger alongside NBI torque and raw kHz Mirnov.
4. The Mirnov statement becomes a sampling-theorem derivation with §8b.2 as its confirmation.

Artifacts: `data/.physics_scales.json`; script `ces_prediction/analyze_physics_scales.py`.

**Applied to the paper (same day).** `docs/paper/main_ko.tex` and `main.tex` §4 were rewritten from
53 / 58 lines into **seven subsections** carrying the first displayed equations the paper has ever
had (loss, noise floor, the two conservation laws, Nyquist, `τ_eq`) plus `tab:torque`. The
model-to-results line ratio goes from **11.2 to 2.5** (ko) and 2.4 (en). §4.7 states plainly that
the optimizer, batch and schedule were inherited rather than derived, records the one piece of
evidence that defends them (B.1's budget-equalization arm, fixed 10 epochs, 4/4 positive), and names
the two measurements that would close the gap — gradient noise scale for the batch, `μP` for the
width sweep. Neither has been run; both are listed as pending, and neither touches TEST.

---

## 8as. The gradient noise scale (2026-09-03) — "why batch 16" gets a bounded answer, and the first estimator had to be thrown away

**Question.** §4.7 of the paper now states plainly that the optimizer, batch size and schedule
were inherited from the windowed pipeline rather than derived, and this project's rule (§8j) says a
gap must name the measurement that closes it. It named two: the gradient noise scale for the batch,
and `μP` for the width sweep. This section runs the first.

**Design** (`ces_prediction/experiments/gns/run_gns.py` → `data/.gns.json`). CPU only — a CUDA
device was present but already held by another job. The confirmed recipe exactly (`seq_v2`,
AdamW 1e-3 / wd 1e-4, masked MSE + 0.1 ReLU penalty, clip 1.0, batch 16 blocks), the data treatment
**pinned in the runner** (held-free, 3 keV cut, per-shot standardization), and the frozen B.1 seed-42
manifest read for `train_files` / `val_files` only. **TEST is never loaded and no model is
promoted**, so this needs no pre-registration entry — it measures the optimizer, not the claim.

### 1. The two-batch-size estimator failed, and that is the first result

McCandlish et al. (2018) estimate `B_simple = tr(Σ)/|G|²` from gradients at two batch sizes:

    |G|²      ≈ (B_big·|g_big|² − B_small·|g_small|²) / (B_big − B_small)
    tr(Σ)     ≈ (|g_small|² − |g_big|²) / (1/B_small − 1/B_big)

With `B_small = 2`, `B_big = 32`, 40 and 10 draws per point, over 12 epochs:

| | init | ep1 | ep2 | ep3 | **ep4** | ep5 | ep6 | ep7 | ep8 | ep9 | **ep10** | ep11 | ep12 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `B_simple` | 19.7 | 8.7 | 3.3 | 6.3 | **NaN** | 10.2 | 1.6 | 0.8 | 8.8 | 110.1 | **NaN** | 3.9 | 6.9 |

**`|G|²` came out negative at epochs 4 and 10** (−0.043, −0.065). The estimator subtracts two
sampled gradient norms that become nearly equal once the model has converged and `|G|²` is small,
so the difference is dominated by sampling noise; more draws shift the failure point but do not
remove it. The 0.8–110 range is not a measurement. **Do not report these numbers, and do not use
this estimator on a converged model.**

### 2. Computing it exactly instead of sampling it

The sampling unit is a block (one contiguous segment) — that is what the training loop draws. With
`g_i` the gradient of block `i`'s own masked loss, everything is available in one pass:

    G = mean_i g_i,   tr(Σ) = mean_i |g_i|² − |G|²,   B_simple = tr(Σ)/|G|²

Only a running gradient sum (one parameter-sized vector) and a running scalar are held, so memory is
flat in the number of blocks, and Cauchy–Schwarz keeps `tr(Σ) ≥ 0` by construction. The cost is one
batch-1 backward per block per measurement point (452 blocks), and the result is the **empirical**
noise scale of that checkpoint rather than an estimate of it.

| point | `|G|²` | `mean|g|²` | `tr(Σ)` | **`B_simple`** | noise share |
|---|---:|---:|---:|---:|---:|
| init | 6.481 | 86.46 | 79.98 | 12.34 | 92.5% |
| epoch 1 | 0.2727 | 10.52 | 10.25 | 37.59 | 97.4% |
| epoch 2 | 1.739 | 12.14 | 10.40 | 5.98 | 85.7% |
| epoch 3 | 0.1513 | 9.445 | 9.293 | 61.42 | 98.4% |
| epoch 6 | 0.2804 | 8.238 | 7.957 | 28.38 | 96.6% |
| epoch 9 | 0.2592 | 7.942 | 7.683 | 29.64 | 96.7% |
| epoch 12 | 0.1610 | 7.506 | 7.345 | 45.62 | 97.9% |

Post-init: **median 33.6, geometric mean 28.4, range 6.0–61.4 blocks.**

### 3. What is stable and what is not

`mean|g|²` and `tr(Σ)` fall smoothly and monotonically after epoch 1 (10.5 → 7.5 and 10.3 → 7.3).
**All of `B_simple`'s tenfold spread comes from `|G|²`**, which bounces between 0.15 and 1.74 — and
`|G|²` is computed *exactly* over all 452 blocks, so that bounce is a real property of where the
Adam trajectory happens to be at an epoch boundary, not measurement error. Any `B_simple` for this
problem is therefore an order-of-magnitude statement.

The stable statement is the noise share: after the first epoch, **86–98% of a single block's
gradient energy is noise** and only 2–14% is signal.

### 4. Verdict — the answer §4.7 was missing

**Batch 16 sits at about half the measured noise scale (~30 blocks).** Below `B_simple`, batch size
and step count trade roughly linearly, so:

1. 16 is **neither wasteful nor obviously suboptimal** — it is inside the regime where every extra
   example still buys step-efficiency.
2. Doubling to ~32 would roughly halve the step count at the same data efficiency; going far above
   ~60 would begin to waste examples. That is the bounded statement the paper can now make in place
   of silence.
3. The claim is order-of-magnitude, for the reason in §3.

**What this does not show.** Dropout is off during the measurement, deliberately: `B_simple`
describes the noise from *sampling examples*, and leaving dropout on would fold a second, unrelated
source into `tr(Σ)`. The training loop's own gradient carries both, so the loop's effective noise
scale is **at least** the number above. The per-block loss normalizes by each block's own label
count while the training loop pools the denominator across a batch, so `G` here is the mean of
per-block gradients rather than the label-weighted full-batch gradient — the standard per-example
formulation, and the difference is small when label counts are similar. Nothing here says batch 16
produces a *worse model*; `B_simple` is about the compute/step trade, not about final skill. And
**`μP` remains unrun** — the second measurement §4.7 names, and the one that would settle whether
§8aa's flat width sweep is a capacity statement or a learning-rate one.

Artifacts: `data/.gns.json` (exact estimator), `data/.gns_run.log`; script
`ces_prediction/experiments/gns/run_gns.py`.

---

## 9. Recommended framings for the thesis (rewritten 2026-08-19 after B.9)

**The claim to lead with.** *About 50 ms of contiguous causal context is what makes the win over
the strongest deployable baseline **typical** rather than average-only; how you cross it decides
cost, not skill.*

§8am corrected the earlier wording ("what decides whether the nowcaster beats..."): pooled over 301
discharges the model beats the causal GP at **every** measured context, 20 ms included. What context
buys is generality — the fraction of discharges it wins rises 0.52 -> 0.66 and flattens.

That sentence is a compression of the one contrast B.9 was built to make. Two things were varied
and only one moved the result:

| varied | range | `CES_TI` vs `gp_causal` | significant |
|---|---|---|---|
| **architecture**, reach fixed | LSTM 357,570 / TCN 3,238 / attention 295,746 | +0.113 / +0.119 / +0.119 | 4/4 · 4/4 · 4/4 |
| **context**, architecture fixed | `seq_v2` at 20 ms → 70 ms → 630 ms | **+0.055 → +0.115 → +0.138** | **2/4 → 4/4 → 4/4** |

A 110× parameter range and three of the most dissimilar sequence operators available produce no
distinguishable difference. Fifty milliseconds of context is the difference between missing the
promotion bar and clearing it 4/4.

**Two qualifiers the sentence needs, both measured after it was written.** *Skill, not cost* holds
**above ~10k parameters** (below that the convolutional arm beats the size-matched recurrent one by
+0.027…+0.040, §8ai) and **at 150 ms of context or more** (at the 70 ms rung itself the attention
arm is 0.023 below the same-reach LSTM, 3/4 significant — §8ak). Recurrence and dilated convolution
turn at the *same* 70 ms rung and tie each other at every rung they share, so the threshold itself
is not an artifact of the LSTM. Both qualifiers are small next to the effect being claimed: the
largest family gap anywhere is 0.023 against a reach effect of +0.060.

### 9.1 State the threshold relative to its baseline — it is not "the context needed to be useful"

At 20 ms of context the model already beats persistence by **+0.356**, which is what a deployed
system actually displaces. What 20 ms fails to beat is `gp_causal`, the strongest past-only method
(§8x). **70 ms is the context needed to beat the best available alternative, not the context needed
to be worth deploying.** Say it that way; the weaker phrasing invites the objection that the
threshold is an artifact of an easy baseline, and the stronger one is not what was measured.

**The number is 50 ms and the rule that produces it is §3.4** (smallest reach whose deficit
against full context is under 0.02, ≤ 1/4 significant). It is a **saturation** point, not a crossing
point — §8am shows 20 ms already beats the causal GP (+0.057 [+0.027, +0.085] pooled). §8al densified the ladder to one-step
spacing; 70 ms was an artifact of never having trained rungs 4, 5 and 6. Quote "≈ 50 ms".

Do **not** quote "the first rung reaching 4/4 against `gp_causal`" as the threshold. That count
flickers by ±1 between adjacent rungs (4/4 at 40 ms, 3/4 at 50, 4/4 at 60 and 70, 3/4 at 100)
because it is a five-level vote on four splits, while the point estimates underneath rise smoothly.
Report the counts, decide on the deficit.

### 9.2 Two resources, not one — this reconciles §8f with §8af

§8f found the *window* sweep flat from W = 2 to W = 8 and §8af finds a 70 ms threshold. They do not
conflict, because they count different things:

- **past CES observations** — one is enough (§8f, unchanged);
- **contiguous fast-diagnostic context** — ~50 ms is needed (§8af, corrected by §8al).

The window family satisfied the first and could not reach the second *at any W*, by construction:
an `AdaptiveAvgPool1d(1)` averages the window and discards its ordering (§8ad), the temporal-subset
augmentation makes each window an arbitrary non-contiguous subset of past rows, and rows without an
observed target were dropped before the model ever saw them. The numbers land exactly where that
explanation predicts — window `iter009` at W = 2 scores **+0.041 (1/4)** against `gp_causal` and
`seq_v2` truncated to the same 2 steps scores **+0.055 (2/4)**. Different architectures, same
starvation, same result.

So the full-grid reframing (§8d, §8t) was not a modelling preference; it was the only way to make
the resource that matters reachable. **That is the paper's mechanism story, and it retroactively
explains why §8f was flat.**

### 9.3 Choose the architecture on cost, because skill does not distinguish it

Above a modest capacity **and above 150 ms of context** the operator is irrelevant to skill (§8ag,
bounded by §8ai below ~10k parameters and by §8ak at the 70 ms rung), so the selection is made
entirely on price — and the price is derivable from structure, not from parameter count or arithmetic. **Quote
the operator count, not the milliseconds**: §8aj measured cost as `N_ops × ~2–3 µs` and the counts
are exact on any machine, while this machine's absolutes moved by 21.84× between sessions.

| family | cost vs reach R | `aten::` ops per online step | at R = 15 | at R = 63 |
|---|---|---|---:|---:|
| recurrent (LSTM) | **O(1)** — the state carries it | 111 fused / 161 lean, at *every* reach | 111 | 111 |
| dilated causal conv | **O(log R)** | 113 + 48·(L−1), `R = 2^(L+1) − 1` | 209 | 305 |
| banded attention | **O(1)** — the KV cache absorbs it | 473 fused / 565 lean, at *every* band | 473 | 473 |

Attention is *also* O(1) in reach — 565 ops at bands 7, 15 and 63 alike — it simply carries a
**3.5–4.3× worse constant**. On this problem it is **strictly dominated**: `tcn3k` matches its
skill (+0.119, 4/4) with 91× fewer parameters and 2.3–2.7× fewer dispatched operators. The conv
column above is the shared lean rewrite; the packed tight step (§8aj addendum, 2026-08-21) halves
it — `tcn2k_tight` runs at **106 dispatches**, below the fused backbone's 111.

**Two operating points are worth naming.** `tcn3k` (3,238 parameters) matches the 357,570-parameter
backbone on `T_i` and beats `gp_causal` 4/4; `v2m4k` (3,898) is the cheapest recurrent arm that
still clears 4/4, and `tcn2k` (1,808) is the cheapest arm of any family (§8ai). Neither is "the
small version of the model" — they are the answer to what the problem actually requires. On this
machine their minimum median steps were 0.45 / 0.19 / 0.52 ms, but those are lower bounds from a
noisy pass and belong in a caption, not in a claim.

### 9.4 Report the `T_i` / `V_rot` asymmetry as a mechanism, and attribute the gap to data

`V_rot` is the honest weak side, and §8al identified *why* — which changes the sentence. **No arm
reaches 3/4 against `gp_causal`**, but that is **not** a power ceiling: the model wins on only
**48% of discharges**, the median discharge's skill is ≈ 0, and removing five of ~62 shots takes
the pooled advantage to zero or below on all four splits (top-1 share 49–164%). The `T_i` control
is the opposite — 70% of discharges, top-1 share 9–16%.

So write: **the `V_rot` advantage is not shot-general — large wins on a minority of discharges,
losses on the rest.** Do not write "underpowered at ~96 discharges"; more discharges do not fix a
coin-flip win rate, and saying so promises a fix that does not exist. Do not call it a failure of
reconstruction either — every arm beats persistence by ≈ +0.39.

**And give the mechanism, because §8an measured it.** The win concentrates where the quantity
*moves*: by tercile of the truth's spread within a discharge, `V_rot` wins 34% / 48% / 55%, and
`T_i` wins 42% / 83% / 85%. Quiet discharges are lost because a flat line is what a GP is best at.
That explains the quiet half — and the fact that `V_rot` reaches only 55% even in the variable third
is the same finding as §2.3d from the other side: **the driver of rotation is unobserved**, so the
model cannot track motion it has no input for.

The asymmetry has a mechanism and it now shows up in a second coordinate. `V_rot`'s best family is
the small **recurrent** arm and `T_i`'s is the small **convolutional** one, which is what §8ab's
routing predicts: `V_rot` rides a highly autocorrelated carried value, so recurrence suffices, while
`T_i` integrates fast-diagnostic history, where weight sharing across time pays.

Attribute the residual gap to **named, testable data levers**, never to model inadequacy (§8j):
NBI torque is absent from the dataset (`T_e`~`CES_VT` r = +0.024, p = 0.58, against
`T_e`~`CES_TI` r = +0.353, p = 3e−17), and the Mirnov stream — the one plausible rotation proxy —
was destroyed by unfiltered decimation (lag-1 autocorrelation −0.009 vs BES +0.568). Both are
acquisition tasks, and the torque one is not fixed by a higher sampling rate.

### 9.5 Frame the microsecond programme as an extension, not a dependency

The thesis is complete on the 10 ms grid. Keep the µs work in one forward-looking section and be
precise about what it can and cannot do:

- **It cannot produce µs targets.** CES integration time is ~10 ms, so `T_i` and `V_rot` labels stay
  on the 10 ms grid no matter what is acquired. The µs question is whether **higher-bandwidth
  input** improves a 10 ms prediction (B.6), not whether prediction moves to µs cadence.
- **It is where non-uniform sampling first appears.** The present grid is 99.46% uniform
  (Δt = 0.01 s, three distinct values), so "handles irregular sampling" is not a claim this data
  can support. Multi-rate acquisition is what makes it real.
- **It turns attention's defeat into a prediction.** Attention is the only family whose reach is
  defined in *elapsed time* rather than in *steps* — a recurrent transition and a dilated kernel
  both assume a uniform grid by construction. So the arm that loses here is the one predicted to
  win when the grid stops being uniform, which is a falsifiable statement rather than a consolation.

### 9.6 What not to claim

- Not "beats every offline method": the model **ties** the acausal GP (§8p, §8ac).
- Not "the window framing is refuted on information" (§8ac's original wording): 87% of that deficit
  was cold start, and a model *trained* at reach 2 recovers it (§8ae, §8af).
- Not a 1 ms deployment verdict from this hardware: session-to-session p99 spread reached **21.84×**
  and the pre-registered protocol correctly refused to rule. Minimum latencies are robust (noise
  only adds) and support "every trained arm's step is under a quarter of a millisecond"; the p99
  budget test needs a controlled machine.
- Not "more context is better": the curve is a **threshold**, flat past 70 ms, and unbounded context
  is slightly *worse* than a bounded 630 ms (§8af).

---

---

## 8at. Is the CES missingness a beam-modulation duty cycle? (2026-09-05) — no, and that narrows a claim made the same day

**Where the question came from.** A second literature sweep (prompted by 승상님: the first pass had
asked only for "prediction / virtual diagnostic / super-resolution", which is not what the fusion
community calls this work) found **NN-CES** — J.K. Lee, W.H. Ko, H.H. Lee, B. Kim, Y. Lee, G. Shin,
J. Kim, J.M. Lee, *Neural network-based analysis of charge exchange spectra in KSTAR*, Fusion
Engineering and Design **222**:115518 (2025), doi:10.1016/j.fusengdes.2025.115518, open access. It is
the closest published work to this thesis: a physics-constrained network trained on ~150,000 KSTAR
frames (L-mode, H-mode, locked mode, RMP) that reads `T_i` and toroidal rotation straight out of the
CES spectra, with a loss combining parameter error and spectral-shape consistency.

Its abstract states the instrument fact that matters here: **KSTAR runs CES off the main heating
beam, modulated**, rather than off a dedicated diagnostic beam, and NN-CES removes the need for that
modulation. The obvious inference — *then our gaps are the modulation duty cycle, and a
modulation-free analysis would delete them at the source* — was written into the appendix within the
hour. This section measures it instead of asserting it.

**Design** (`ces_prediction/analyze_ces_duty_cycle.py` → `data/.ces_duty_cycle.json`, all 641 files,
247,207 rows, 667 contiguous 10 ms blocks; descriptive only, no model, no split, no TEST).
Four statistics, each chosen to separate *one shared shutter* from *per-channel outcome*:
run lengths of observed and missing stretches per target; the dominant FFT period of each
observation mask; the agreement between the `T_i` and `V_rot` masks, computed **twice** (raw NaN
masks, and after the confirmed protocol's 3 keV cut); and the short-lag autocorrelation of each mask.

### 1. Coverage and mask agreement

| quantity | value |
|---|---:|
| both targets observed | 70.0% |
| `T_i` only | 21.4% |
| `V_rot` only | 6.1% |
| neither | 2.5% |
| **mask agreement, raw** | **72.8%** |
| mask agreement, after the 3 keV cut | 72.5% |

**This is the decisive row.** A single beam-modulation shutter gates the spectrometer, not one
channel of it, so it would drive the two masks to agree essentially always. They agree
**72.8%** of the time *before* our treatment touches anything, and our own 3 keV cut moves
that by **0.3 percentage points** — nothing. So the disagreement is in the delivered
data, and the two channels go missing largely independently, exactly as the data contract already
assumed.

### 2. Run lengths and periodicity

| statistic | `T_i` | `V_rot` |
|---|---:|---:|
| missing runs, median / mean / p90 | 1 / 2.81 / 5 | 1 / 18.83 / 11 |
| observed runs, median / mean | 2 / 29.75 | 3 / 55.67 |
| dominant FFT period, median | 1480 ms | 2980 ms |
| mask autocorrelation, lag 1 → 10 | 0.327 → 0.123 | 0.299 → 0.211 |

`T_i` gaps are overwhelmingly **isolated single frames** — 4,028 of 7,169 missing runs are
one 10 ms step. The dominant periods sit at the block-envelope scale (1.5 s, 3.0 s), not at any duty
cycle, and the short-period modes (2–3 steps) appear in only ~20 of 667 blocks. The
autocorrelations **decay smoothly with no ripple**; a fixed modulation period would oscillate.

### 3. Verdict, and the claim it narrows

**No duty cycle is visible in our data.** The rows we hold are already the background-subtracted,
fitted 10 ms product, so whatever modulation KSTAR ran has been folded in upstream; what remains
missing is a per-channel analysis outcome, not a beam gate we could see.

The same-day appendix sentence "빔 변조가 필요 없다는 점은 결측 자체를 줄일 수 있다" is therefore
**demoted from a mechanism to a request**. What NN-CES credibly offers this work is not fewer holes
in the file we already have; it is (i) **per-frame fit quality**, which is the named replacement for
our value-based 3 keV cut and would collapse the two co-primary populations into one (§8q, §8ab), and
(ii) the possibility of a **denser CES product re-derived from the raw spectra**, which is a data
request to KFE — W.H. Ko, a co-author, is the KSTAR CES diagnostic lead — and not something our masks
can establish.

**What would overturn this.** The raw CES frame timing, or the beam-modulation log for these 641
discharges. If the modulation period is present upstream at a resolution finer than our 10 ms grid,
this measurement cannot see it, and says so.

**Positioning, stated once so it is not restated loosely.** NN-CES reads the instants where CES
measured; this work fills the instants where it did not. Its inputs are the CES spectra themselves,
so it cannot run where there is no frame. That distinction is now on the related-work slide.

Artifacts: `data/.ces_duty_cycle.json`, `ces_prediction/analyze_ces_duty_cycle.py`.

## 8au. Which information routes are open to a `V_rot` imputer? (2026-10-07) — the cross-target route is closed, the shot-level and unlabeled-row routes are open

**Where the question came from.** 승상님 asked to move the thesis's validation away from
hypothesis-test framing toward a comparison of multimodal model families (diffusion imputation,
shot encoders / neural processes, masked pretraining, cross-modal transformers), with `V_rot` as the
motivation. Before building any of them, this section measures whether the information each family
would lean on is present at all. Plan: `experiments/PREREGISTRATION_B12.md` (v0 proposal).

**Design** (`ces_prediction/analyze_vrot_info_routes.py` → `data/.vrot_info_routes.json`, all 641
files, held-free, both populations; descriptive only, no model, no split, no TEST).

| route | statistic | cut 3 keV | inclusive |
|---|---|---:|---:|
| cross-target | `V_rot` gaps where `T_i` is observed at the same instant | 0.909 | 0.914 |
| | Pearson / Spearman `T_i`–`V_rot` (co-observed rows) | 0.024 / 0.049 | 0.016 / 0.049 |
| | η² of `V_rot` on 20 quantile bins of `T_i` (any function) | **0.0020** | **0.0019** |
| | same, shot-demeaned | 0.0017 | 0.0013 |
| shot level | share of `V_rot` variance that is between-shot | **0.281** | 0.281 |
| | shots with 0 / 1–20 / > 20 `V_rot` labels | **65 / 202 / 374** | 65 / 202 / 374 |
| label scarcity | rows with complete inputs but no `V_rot` label | **0.650** | 0.650 |

**Verdict.**
- **Cross-target: closed.** `T_i` is observed in 91% of the `V_rot` gaps, so a joint generative
  imputer would have that input almost everywhere, but no function of `T_i` explains more than 0.2%
  of `V_rot` variance, either pooled or within a shot. A diffusion model over `[T_i, V_rot]` can't
  get `V_rot` skill out of `T_i`. B.12 therefore puts diffusion on the probabilistic axis, not on
  `V_rot` point skill.
- **Shot level: open.** 28% of `V_rot` variance is a per-shot offset, and 267 of 641 shots (42%)
  have at most 20 `V_rot` labels, so same-shot interpolation has little to work from there. This is
  the route a shot latent (neural process) could use, and it was never tested: every model so far
  uses information step by step.
- **Unlabeled rows: open.** 65% of input rows carry no `V_rot` label. That is what masked
  pretraining would use.

**What would overturn this.** η² measures dependence at the same instant only. A lagged dependence
(`T_i` leading `V_rot` through the shared NBI drive) is not measured here; the `diff` arm's
time-axis attention would pick it up, which is why the arm stays in B.12 on the probabilistic axis.

Artifacts: `data/.vrot_info_routes.json`, `ces_prediction/analyze_vrot_info_routes.py`.

## 8av. B.12 offline benchmark, baselines only (2026-10-08) — on `V_rot` gaps, same-shot interpolation cannot be told apart from the training mean

**What this is.** The B.12 offline gap-filling benchmark (`experiments/PREREGISTRATION_B12.md`, v0) scored with
its four model-free arms before any network trains. Observed CES rows of the test shots are hidden in spans
drawn from the train files' own missing-run-length distribution (20% of each file's observed rows per target),
and every arm fills the same fixed points. Context = the remaining observed rows of the same block. Split
manifests are B.1's (`data/.b1_manifest_s{42,1,7,123}`), held-free, both populations. Shot-clustered bootstrap,
pooled over the four splits. Code: `experiments/b12/{b12_data,baselines_b12,card_b12}.py`; artifacts
`data/.b12_masks/`, `data/.b12_base_cut*_s*/`, `data/.b12_card.json`.

Skill vs PCHIP (`1 − MSE/MSE_PCHIP`, pooled, 95% CI):

| population / target | n | train mean | acausal GP | GP per split (42/1/7/123) |
|---|---:|---|---|---|
| cut `T_i` | 28,623 | −1.94 [−2.54, −1.47] | **+0.223** [+0.15, +0.30] | +0.29 / +0.17 / +0.09 / +0.32 |
| incl. `T_i` | 28,743 | +0.05 [−0.15, +0.23] | **+0.330** [+0.22, +0.44] | |
| cut `V_rot` | 18,297 | +0.02 [−0.59, +0.32] | +0.044 [−0.00, +0.12] | −0.01 / +0.09 / +0.02 / +0.17 |
| incl. `V_rot` | 16,855 | +0.03 [−0.50, +0.31] | +0.049 [+0.01, +0.12] | +0.09 / +0.17 / −0.01 / +0.08 |

Label-sparsified `V_rot` sets (dense test shots keeping only k labels; identical in both populations):
k = 3 → GP +0.127 [+0.02, +0.30], mean −0.26 [−0.70, −0.02]; k = 10 → GP +0.079 [+0.01, +0.19], mean +0.03 [−0.66, +0.42].

**Reading.**
- `T_i` behaves as §8p said it would offline: GP beats PCHIP 4/4 in both populations, and the
  ordering mean ≪ PCHIP < GP is clean.
- **`V_rot` does not.** On hidden `V_rot` points, three quarters of which sit in gaps longer than 10 steps,
  PCHIP's pooled MSE can't be separated from the global training mean, and the GP's margin over PCHIP is
  +0.04–0.05 with the cut-population interval touching 0. Same-shot `V_rot` labels carry little that a
  constant doesn't already carry once a gap is long. This is the room a shot-level model would have to fill (§8au): it can only
  win where the same-shot route is weak.
- **Design consequence, fixed before any arm is scored**: the real 1–20-label stratum has only 67–81 hidden
  `V_rot` points pooled, so H-np is tested on the label-sparsified sets instead (PREREGISTRATION_B12.md §3).

**What would overturn the `V_rot` reading.** The mean-vs-PCHIP interval is wide because a few shots dominate the
squared error. A per-shot normalized score (each shot's MSE over its own `V_rot` variance) would show whether the
tie is general or carried by those shots. The card's axis 3 adds it when the arms report.

*All quantitative claims above were regenerated on 2026-07-14 from the checkpoints listed in the
provenance table, using the repository's own harness (`compare_baselines.py`, `bootstrap_compare.py`,
`peak_analysis.py`, `evaluate.py`) with the architecture pinned as described. The model neither
denormalizes internally nor was tuned for this report.*

## 8aw. B.15 deep ensemble, stage 1 (2026-10-11) — the 4-init ensemble is +0.02 on `T_i`, 4/4 significant, and lands exactly on the tie rule

**Question.** The only lever on record with a measured positive sign was a 4-member
init-seed ensemble (+0.048 `T_i`, 2026-07-27, W = 4, reduced config). It had never been
re-checked under the confirmed protocol, and the paper's headline is a single run. Rules
were frozen before any number in `experiments/PREREGISTRATION_B15.md`, with the prior
expectation written down as "+0.01…+0.02, at the tie rule", because the B.1 grid's init
spread (0.009–0.027 per split) is a tenth of the spread where +0.048 was measured.

**Design** (`experiments/b15_ensemble/run_b15.py`; artifacts `data/.b15_ens4_cut_s*`,
verdict `data/.b15_summary.json`). Members = the frozen B.1 grid `.b1_seqv2_s{s}_i{42,1,7,123}`
(val-only early stopping, no TEST peeking); predictions averaged in normalized space
(the members share the split's train files, hence its stats — asserted before scoring);
`eval_seq.py`'s new `CES_SEQ_ENSEMBLE_DIRS` path, unset → bit-identical to every frozen
artifact. Paired against the canonical run (init = split seed, the headline run) and
against each member; PR4 ladder from `bootstrap_compare.py`. **No training.** Cut
population; the inclusive population is stage 2 (12 new inits, below when it finishes).

| split | paired `T_i` (ens4 − canonical) | vs each member (min) | `T_i` vs PCHIP ens4 / canonical | `T_i` vs causal GP ens4 / canonical | paired `V_rot` |
|---|---|---|---|---|---|
| 42 | **+0.011** [+0.004, +0.019] | +0.004 | +0.183 / +0.174 | +0.088 / +0.078 | +0.005 [−0.005, +0.014] |
| 1 | **+0.010** [+0.003, +0.021] | +0.001 | +0.256 / +0.248 | +0.142 / +0.133 | +0.005 [+0.000, +0.013] |
| 7 | **+0.027** [+0.020, +0.036] | +0.000 | +0.277 / +0.257 | +0.161 / +0.138 | +0.000 [−0.007, +0.005] |
| 123 | **+0.030** [+0.009, +0.060] | −0.006 | +0.287 / +0.264 | +0.132 / +0.105 | +0.002 [−0.008, +0.005] |

Mean paired `T_i` **+0.0196**, run-cluster bootstrap CI [+0.010, +0.029], **4/4
individually significant**; `V_rot` significant deficits **0/4** (and no significant
gain either). Every PR4 and causal-GP verdict of the canonical runs is unchanged by
ensembling (`T_i` 4/4 + 4/4, `V_rot` vs PCHIP still 1/4).

**Verdict (pre-registered §4).** The pooled mean is 0.0196 ≤ `PRACTICAL_EPS` = 0.02, so
by the rule written before the number this is a **tie**, and the deployment
recommendation is **not** granted. Read it exactly: the effect is real (4/4 significant,
CI excludes zero) and consistently positive, but it is smaller than the smallest
difference this project has agreed to act on. Two things follow.

1. **The 2026-07-27 +0.048 was init noise, not model gain.** The ensemble gains
   roughly what the members disagree by; under the confirmed protocol they disagree by
   0.01–0.03 and the ensemble buys ≈ +0.02. The memory note that quoted +0.048 as a
   standing lever is corrected to that reading.
2. **The single-run headline is the deployment number.** The paper's `T_i` skill does
   not need a footnote saying "an ensemble would do better": it would do better by
   +0.02 at 4× the inference cost (≈ 6 ms stateful step from §8ac's 1.49 ms, still inside
   the 10 ms budget, outside 1 ms).

**Observation (reported, not a verdict).** The ensemble beats the *best* member on three
splits and loses to it by 0.006 on s123 — so the winner's-curse argument of 2026-07-27
(the ensemble is the best choice *without* reading TEST) still holds, but the margin it
protects is now 0.00–0.03 rather than 0.05.

**Stage 2 — inclusive population (2026-10-12).** Three new inits per split were trained
under the B.5 inclusive env (`.b15_member_incl_s{s}_i{i}`, 12 GPU runs, each TEST-scored
once) and averaged with `.b5i_seqv2_s{s}`. Member `T_i` skill vs PCHIP spreads 0.007–0.020
per split (42: 0.218–0.225, 1: 0.219–0.238, 7: 0.291–0.306, 123: 0.313–0.318). Paired
`T_i` (ens4 − canonical) **+0.005 / +0.003 / +0.014 / +0.005**, mean **+0.007**, run-cluster
CI [+0.004, +0.012], 2/4 individually significant; `V_rot` deficits 0/4 (s42 +0.009*).
**Tie**, by the same rule, with a smaller margin than the cut population — the inclusive
members disagree even less. Both populations say the same thing: the ensemble buys
+0.01…+0.02 and the headline run is the deployment number.

Artifacts: `data/.b15_ens4_{cut,incl}_s{42,1,7,123}/` (`comparison_errors_test.npz`,
`bootstrap_summary.json`, `paired_vs_canonical.json`, `paired_vs_member_i*.json`),
`data/.b15_member_incl_s*_i*/`, `data/.b15_summary.json`.

---

## 8ax. B.13 robust co-metrics (2026-10-12) — the headline skill is a **tail** effect: in the median row the nowcaster is worse than PCHIP and no better than persistence

**Question.** §8aq measured that 47–85 % of the target's own squared 10 ms scatter sits
in 1 % of the rows and named this re-scoring as the next measurement: is the backbone's
`T_i` +0.24 a bulk effect or a tail effect, and is `V_rot`'s "n.s. vs PCHIP" a tail
artifact? Rules, metrics and the sentences each outcome would write were frozen in
`experiments/PREREGISTRATION_B13.md` before any number existed.

**Design** (`experiments/b13_robust/rescore_b13.py` → `data/.b13_summary.json`). The frozen
TEST artifacts only — cut `.b1_seqv2_s{s}_i{s}` / `.b1_w2cut_s{s}`, inclusive `.b5i_seqv2_s{s}`
/ `.b5i_w2_s{s}` — with `|e| = √se`; no training, no inference. Guards: the MSE skill vs PCHIP
recomputed from every npz equals the recorded value (to 1e-5, float32 vs float64 reduction),
and every shot id maps onto its split's test files. Metrics: M1 median-AE skill
(weighted-median shot-cluster bootstrap, B = 10 000), M2 Huber skill at the §8aq order-4
noise bound (129.9 / 274.3 eV, 16.4 km/s), M3 MSE skill after a symmetric 1 % trim on
`max(se_a, se_b)`, M4 the trimmed rows' SSE share, M5 the `|e|` quantile ratio ladder.
Pooled = 4 splits, cluster = physical shot (301 shots).

### 1. The numbers that overturn the reading

Pooled, backbone vs PCHIP (`*` = CI excludes 0):

| population / target | MSE skill (headline) | **M1 median-AE skill** | M2 Huber | M3 trimmed 1 % | M4 tail SSE share model / PCHIP |
|---|---|---|---|---|---|
| cut `T_i` | +0.243* | **−0.198** [−0.259, −0.135] | +0.156* | +0.210* | 0.48 / 0.51 |
| cut `V_rot` | +0.238* (per split 1/4) | **−0.488** [−0.604, −0.387] | −0.060 | −0.212 | 0.84 / 0.90 |
| incl `T_i` | +0.278* | **−0.562** [−0.703, −0.432] | +0.130* | +0.057 | 0.83 / 0.87 |
| incl `V_rot` | +0.238* (2/4) | **−0.518** [−0.630, −0.414] | −0.056 | −0.200 | 0.85 / 0.90 |

Per split, M1 vs PCHIP is negative on **16/16** (population × target × split) cells, and
significantly so on 16/16. In physical units, pooled median absolute error:

| population / target | backbone | PCHIP | acausal GP | causal GP | persistence | window (W = 2) |
|---|---:|---:|---:|---:|---:|---:|
| cut `T_i` [eV] | **47.2** | 39.4 | 38.5 | 44.4 | 46.1 | 52.6 |
| cut `V_rot` [km/s] | **4.3** | 2.9 | 2.9 | 3.8 | 3.7 | 4.9 |
| incl `T_i` [eV] | **62.2** | 39.8 | 39.7 | 45.5 | 46.5 | 65.0 |
| incl `V_rot` [km/s] | **4.4** | 2.9 | 2.9 | 3.8 | 3.7 | 5.6 |

The quantile ladder says where the two readings cross (cut `T_i`, ratio model / PCHIP):
q50 **1.20**, q60 1.11, q70 1.01, **q75 0.96**, q90 0.86, q99 0.78. Against persistence the
crossover is q55 (1.00), against the causal GP q65 (0.99). The model beats PCHIP on
**48 %** of `T_i` rows and **40 %** of `V_rot` rows (incl: 42 % / 39 %); it beats persistence
on 53 % / 47 %.

### 2. Reading

1. **Q2 (`T_i`) = tail, by the pre-registered rule, in both populations.** The +0.24 /
   +0.28 MSE skill is earned above the 70th percentile of error. In the median row the
   backbone's error is 47 eV — **the target's own bulk scatter (§8aq MAD 46.4 eV)** — and
   PCHIP's is 39 eV. The ordering of the bulk column is exactly the ordering of how much
   *noise averaging* each predictor does: an interpolator averages two noisy neighbours
   (GP 38.5 ≈ PCHIP 39.4), the causal GP averages several past samples (44.4), persistence
   copies one noisy sample (46.1) — and the backbone sits **with persistence** (47.2). In
   the bulk the nowcaster carries the last noisy value and adds nothing; it does not
   denoise its own anchor. Huber (M2) and the 1 % trim (M3) stay positive because the
   model's advantage begins well inside the 1 % tail (q75 onward), so "tail" here means
   "the upper quarter of errors", not "a handful of spikes".
2. **Q1 (`V_rot`) is not satisfied — and the sign is the opposite of the hypothesis.** The
   aggregate near-tie vs PCHIP is not a tail artifact hiding a bulk win; it is a bulk
   *loss* (median 4.3 vs 2.9 km/s, −49 %) offset by a tail win (q99 ratio 0.98, peak-stratum
   MSE skill +0.54…+0.79 on 8/8 cells). In the median row the backbone is worse than
   **persistence** (3.7 km/s) and worse than the causal GP (3.8). For a quantity whose
   inputs carry no information (§8ar), the MSE-trained head hedges, and hedging costs the
   median. This is the measurement §8ab's "unresolved" needed: `V_rot` is **resolved
   negative in the bulk** and positive only where the target moves (§8an's covariate).
3. **The inclusive population is worse in the bulk, not just in the tail.** Incl `T_i`
   median 62 eV vs 47 cut: training on uncut spikes degrades the typical row by 15 eV
   while the MSE skill rises (+0.278 vs +0.243). The two-population rule (§1.1) was adopted
   for the tail; it turns out to matter for the bulk too.
4. **The sequence backbone's margin over the window model is a bulk effect** (model vs
   window M1 +0.104* / +0.133* cut, +0.044 / +0.224* incl) — the reach narrative of §8ac
   is about the typical row, which the W = 2 control gets wrong by 5–6 eV more.
5. **The strongest offline arm is unchanged**: the acausal GP has the lowest median in
   every cell and the backbone's M1 vs GP is −0.23 / −0.46 (cut). §8p's "the model ties the
   GP" was an MSE statement; in the median the GP is clearly better.

### 3. What this changes, and what it does not

- **The claim stays; its sentence changes** (pre-registered §5, Q2 tail): "`T_i` 우위는 꼬리에서
  온다. 오차 상위 사분위에서 모델이 PCHIP를 이기고, 중앙값 행에서는 PCHIP가 20 %, persistence가
  2 % 더 낫다." The PR4 headline is unchanged (it was always the MSE); B.13 is the
  co-metric beside it, and the paper's model section gets the physical-unit median table.
- **Headroom is now located.** §8aq said the bulk had 2–3× headroom; B.13 says what the
  model does there — behaves as persistence — and names the mechanism: an MSE loss
  dominated by rows it cannot fit teaches nothing about the rows it could. **B.14 (robust
  loss, already running) is the direct test**, and its M1 column is now the primary
  readout, not a co-metric.
- **A model-side `V_rot` fix is named for the first time.** The bulk loss vs persistence
  (−17 % median) is a hedging cost, so shrinking the `V_rot` head toward its carried anchor
  (the lit-survey axis "soft gate + persistence shrinkage") would recover the bulk without
  new information; the tail win would have to survive. That is a controlled experiment,
  not a claim.
- **Not changed**: PR4 verdicts, populations, the backbone, the causal-GP comparison in
  MSE (4/4), and the §8ar conclusion that `V_rot`'s information is absent. The median
  result is *consistent* with absence: a head without information should not beat
  persistence in the typical row, and it does not.

**Strata (M1 vs PCHIP, per split).** Peak rows: `T_i` −0.02…−0.10 (MSE +0.45…+0.61), `V_rot`
−0.24…−0.58 (MSE +0.54…+0.79) — even where the model wins the MSE by half, its *median*
error is not below PCHIP's. Non-peak: `T_i` −0.16…−0.25, `V_rot` −0.36…−0.73. The win is
concentrated in the large-error rows *within* the peak stratum, not in typical peak rows.

**What would overturn this.** A robust-loss or anchor-shrunk arm whose M1 vs PCHIP is ≥ 0
on 3/4 splits while its MSE skill stays within `PRACTICAL_EPS` of the backbone (B.14's
rule 1). If no training change moves the median below persistence's, the bulk is the
measurement noise of the carried anchor and the fix is denoising the history, not the
head.

Artifacts: `data/.b13_summary.json`, `data/.b13_rescore.log`;
script `experiments/b13_robust/rescore_b13.py` (takes `--extra name=dir` for B.14 / B.15 arms).

---

## 8ay. B.14 noise-floor-pinned robust loss (2026-10-12) — Huber moves the median 8–10 % at zero MSE cost in the cut population, and the MSE early-stop criterion strands it in the inclusive one

**Question.** §8ax located the bulk headroom: the MSE-trained backbone carries its noisy
anchor in the typical row (median 47 eV = persistence's 46 eV), because a loss that 1 %
of the rows decides teaches nothing about the other 99 %. B.14 changes one thing, the
loss's influence function, with the scale pinned to the target's own 10 ms reproducibility
(§8aq order-4 bound: 129.9 / 274.3 eV, 16.4 km/s), and asks whether the median moves
without the aggregate paying. Rules in `experiments/PREREGISTRATION_B14.md`, frozen before
any number; the one amendment (normalizing both losses to coincide with `r²` inside the
scale, so the three arms share the bulk curvature and the penalty ratio) was committed
before the first run, with the reason.

**Design** (`experiments/b14_loss/run_b14.py`; `train_seq.py` `CES_SEQ_LOSS` ∈ {mse, huber,
tnll}; arms `data/.b14_{huber,tnll}_{cut,incl}_s{s}`, MSE summary `data/.b14_summary.json`,
co-metrics `data/.b13_arms_{cut,incl}.json`). seq_v2, split {42, 1, 7, 123} × init = seed,
both populations, everything else the B.1 / B.5 backbone env (per-shot ON, held-free, AdamW
1e-3, batch 16, patience 6 on **val MSE**, cap 60). Controls = the frozen backbone runs, no
retraining. `huber`: `r²` inside δ, `2δ|r| − δ²` outside (gradient capped at the MSE's at δ);
`tnll`: `νδ² log(1 + r²/νδ²)`, ν = 3 (redescending). Normalized δ = 0.33 / 0.22 (cut), 0.43 /
0.22 (incl). Unit tests `tests/test_robust_loss.py`.

### 1. Cut population — the primary readout is the median (§8ax), the guard is the MSE

Pooled over 4 splits, arm vs the MSE control (paired, shot-cluster bootstrap; `*` = CI excludes 0):

| arm / target | MSE paired (guard) | **M1 median-AE** | M2 Huber | M3 trimmed | per-split M1 CI > 0 | MSE significant deficits |
|---|---|---|---|---|---|---|
| huber `T_i` | −0.004 [−0.017, +0.012] | **+0.081*** [+0.068, +0.094] | +0.043* | +0.053* | **4/4** | 0/4 |
| huber `V_rot` | −0.007 [−0.047, +0.017] | **+0.104*** [+0.078, +0.134] | +0.058* | +0.064* | **3/4** | 0/4 |
| tnll `T_i` | −0.018 [−0.033, −0.001] | +0.071* [+0.056, +0.086] | +0.025* | +0.021* | 3/4 | 1/4 |
| tnll `V_rot` | −0.014 [−0.058, +0.012] | +0.074* [+0.037, +0.112] | +0.053* | +0.054* | 3/4 | 0/4 |

Physical units, pooled (cut):

| `T_i` [eV] | median | q75 | q90 | RMSE |
|---|---:|---:|---:|---:|
| backbone (MSE) | 47.2 | 101.0 | 202.4 | 182.6 |
| **huber** | **43.4** | **93.8** | **193.3** | 183.0 |
| tnll | 43.8 | 94.6 | 196.8 | 184.3 |
| causal GP | 44.4 | 106.7 | 219.8 | 194.3 |
| persistence | 46.1 | 116.6 | 262.7 | 235.9 |
| PCHIP (acausal) | 39.4 | 105.4 | 236.6 | 209.9 |

| `V_rot` [km/s] | median | q75 | q90 | RMSE |
|---|---:|---:|---:|---:|
| backbone (MSE) | 4.3 | 10.0 | 21.4 | 33.8 |
| **huber** | **3.8** | **9.2** | **19.8** | 34.0 |
| tnll | 4.0 | 9.3 | 19.7 | 34.1 |
| causal GP | 3.8 | 9.3 | 20.0 | 36.1 |
| persistence | 3.7 | 9.3 | 20.9 | 43.1 |
| PCHIP (acausal) | 2.9 | 7.8 | 18.3 | 38.8 |

Against the ladder (cut, pooled M1): huber vs persistence **+0.058*** (`T_i`, 3/4 significant —
the first arm of this project whose *median* beats persistence), vs the causal GP **+0.023**
[−0.009, +0.055] (tie in the median; the MSE control was −0.063 there), vs PCHIP −0.101 (the
gap to the acausal interpolator is halved from −0.198). `V_rot`: vs persistence −0.047, vs
causal GP −0.018 (both near tie; the control was −0.168 / −0.136), vs PCHIP −0.333.

### 2. Inclusive population — the selection rule strands the robust arms

| arm / target | MSE paired (guard) | MSE significant deficits | epochs run / best (per split) | M1 median-AE |
|---|---|---|---|---|
| huber `T_i` | −0.028 / −0.022 / −0.035 / −0.032 | **2/4** | 10/4, 8/2, 8/2, 23/17 | **+0.157*** [+0.120, +0.196] (3/4) |
| huber `V_rot` | −0.014 / −0.009 / −0.041 / −0.018 | 1/4 | | −0.034 [−0.106, +0.027] (1/4) |
| tnll `T_i` | −0.052 / −0.030 / −0.040 / −0.059 | **3/4** | 10/4, 8/2, 8/2, 8/2 | +0.084* [+0.049, +0.118] (1/4) |
| tnll `V_rot` | −0.019 / −0.012 / −0.044 / −0.066 | 2/4 | | −0.150 [−0.233, −0.077] (0/4) |

Seven of the eight inclusive runs stopped at **epoch 8–10 with the best val MSE at epoch 2–4**
(the cut runs trained 20–46 epochs). The trace is flat, not noisy: huber s1 val MSE
0.490 → 0.470 at epoch 2, then 0.47–0.48 for six epochs. A loss that ignores > 274 eV
residuals has no reason to lower a val MSE that those residuals dominate (§8aq: 85 % of the
inclusive `T_i` squared mass in 1 % of the rows), so the MSE-based checkpoint rule — kept
on purpose so that all three arms were *selected* identically — picked a nearly untrained
model. Even so, the median improved: huber `T_i` 52.4 eV vs the control's 62.2, `V_rot` 4.5
vs 4.4; the MSE cost is the spikes the robust arm declines to fit. Physical medians:
`T_i` huber 52.4 / tnll 56.9 / control 62.2 / causal GP 45.5 / PCHIP 39.8; `V_rot` 4.5 / 5.0 / 4.4.
Pooled, the `T_i` median gain is *larger* than in the cut population (+0.157 vs +0.081: the
inclusive control's median is the worst of all arms, 62 eV, because the spike rows it is
trained to chase pull its typical row off the anchor), while the `V_rot` median is a tie
(−0.034) and tnll's `V_rot` median is significantly worse (−0.150). So in the inclusive
population huber improves the typical `T_i` row by 16 % and pays 0.03 of MSE skill for it;
the pre-registered rule does not trade one for the other, and neither do we here.
(For the record, the same scoring puts the B.15 inclusive ensemble at M1 +0.040* `T_i` /
+0.061* `V_rot` — the ensemble's small MSE gain is also a bulk gain.)

### 3. Verdict (pre-registered §4, unchanged)

- **Rule 1 (robust rung) holds in the cut population only** — huber: H14-a 4/4 + 3/4, MSE
  deficits 0/4 on both targets. The inclusive population fails the MSE guard (2/4 `T_i`
  deficits, pooled −0.030 [−0.046, −0.013]) and H14-a on `V_rot` (M1 1/4), so the rung is
  **conditional on the cut population** and is recorded that way. tnll satisfies
  H14-a but loses the `T_i` MSE guard once (1/4) and is strictly behind huber on every
  metric: **H14-d's ordering is reversed** — redescending influence gives up useful tail
  signal.
- **Rule 2 (replace the backbone's loss) is not met**: `T_i` MSE paired sign is 2/4, not 4/4.
  The backbone keeps the MSE loss; huber is the reported robust rung.
- **H14-c (`V_rot` spikes contaminate training) is not supported**: `V_rot` MSE skill is
  unchanged (−0.007), with 1/4 significant gains and 1/4 deficits across the arms. The
  `V_rot` median improvement (+0.10, 3.8 vs 4.3 km/s) is a hedging correction, not new
  information — the huber `V_rot` median equals the causal GP's and persistence's.

### 4. What this settles, and the measurement it names

1. **The bulk headroom is real and the loss reaches part of it.** 47.2 → 43.4 eV at zero
   aggregate cost; the remaining 4 eV to PCHIP's 39.4 is the noise-averaging advantage of
   seeing the future neighbour, which no causal method can take. In the median the huber
   backbone now ties the causal GP (44.4) and beats persistence; in the aggregate it keeps
   the +0.24. That is the deployable operating point, stated with both numbers.
2. **The inclusive failure is a selection-rule interaction, not a loss failure.** The
   pre-registered fix is a sensitivity row, not a verdict change: re-run the inclusive arms
   with the checkpoint chosen on **val Huber** (the arm's own objective) or at a fixed epoch
   count, and report MSE and M1 both. If the median gain survives and the MSE deficit
   shrinks to the tie band, the rung becomes unconditional; if the MSE deficit persists, the
   inclusive population's spike rows are genuinely fit by the MSE model and the two
   populations disagree about what "better" means — which is the two-population rule's
   whole purpose (§1.1).
3. **Capacity was never the lever; the objective was.** Three flat axes (§8z, §8aa, §8ag)
   and one loss change that moves the median 8 % on every split.

Artifacts: `data/.b14_*`, `data/.b14_summary.json`, `data/.b13_arms_cut.json`,
`data/.b13_arms_incl.json`, logs `data/.b14_batch.log`.

---

## 8az. B.12 model-family card, first three arms (2026-10-12) — the shot-encoding neural process ties the bidirectional control everywhere H-np said it would win, and the MAST transplant brings nothing a random init does not

**Question.** §8au left two information routes open to a `V_rot` imputer: shot-level prior
information (28 % of `V_rot` variance is between shots; 42 % of shots have ≤ 20 labels) and the
unlabeled rows. B.12 asks which inductive bias reaches them, by carding model families against
one offline control that differs from the backbone only in direction (`seq_v2_bi`, here `bilstm`).
This section records the first three arms — `np` (shot latent + cross-attention neural process,
H-np), `tm_pre` / `tm_scratch` (TokaMind transformer with MAST-pretrained vs random backbone,
H-tm) — on card axes 1, 3 and 6 (`PREREGISTRATION_B12.md` §5–§6). `diff`, `mae`, `xmod` are not
started.

**Design** (`experiments/b12/{run_b12,train_b12,b12_models,card_b12}.py`). 4 arms × 4 splits × 2
populations = 32 GPU runs (RTX 5060 Laptop), identical masks, splits, budget (cap 60, patience on
val hidden-MSE), held-free, B.1 manifests; TEST hidden points fixed by §8av and scored once per arm.
Statistic: skill vs PCHIP = `1 − MSE/MSE_PCHIP`; δ vs control = `(MSE_bilstm − MSE_arm)/MSE_PCHIP`;
shot-clustered bootstrap (B = 2000), pooled over 4 splits. §6 rule: δ > 0.02, CI low > 0, same sign
≥ 3/4, *and* the same against the acausal GP. Artifacts `data/.b12_{arm}_cut{3000,0}_s{s}/`,
`data/.b12_card.json`, log `data/.b12_batch.log`.

### 1. Axis 1 — point accuracy, pooled over 4 splits (skill vs PCHIP; δ vs `bilstm` with verdict)

| population / target | `bilstm` (control) | acausal GP | `np` | `tm_pre` | `tm_scratch` |
|---|---|---|---|---|---|
| cut `T_i` | +0.274 [+0.22, +0.33] | +0.223 | **+0.300** [+0.24, +0.36] · δ **+0.026** [+0.004, +0.049] · signs 3/4 · **IMPROVE** | +0.179 · δ −0.094 [−0.13, −0.06] · 4/4 · WORSE | +0.184 · δ −0.090 [−0.12, −0.06] · 4/4 · WORSE |
| cut `V_rot` | +0.364 [+0.08, +0.53] | +0.044 | +0.384 · δ +0.021 [+0.000, +0.050] · 3/4 · IMPROVE (edge) | +0.376 · δ +0.012 [−0.007, +0.035] · TIE | +0.384 · δ +0.021 [+0.001, +0.044] · 3/4 · IMPROVE (edge) |
| incl `T_i` | +0.410 [+0.29, +0.53] | +0.330 | +0.403 · δ −0.006 [−0.017, +0.002] · TIE | +0.371 · δ −0.039 [−0.062, −0.019] · 3/4 · WORSE | +0.378 · δ −0.032 [−0.055, −0.012] · 3/4 · WORSE |
| incl `V_rot` | +0.336 [+0.07, +0.55] | +0.049 | +0.327 · δ −0.009 [−0.042, +0.013] · TIE | +0.321 · δ −0.016 [−0.043, +0.004] · TIE | +0.333 · δ −0.003 [−0.018, +0.012] · TIE |

Per split, δ `np` vs control — cut: `T_i` −0.019 / +0.009 / +0.034 / +0.054, `V_rot` +0.050 / +0.032 /
+0.014 / −0.053; inclusive: `T_i` −0.015 / +0.010 / −0.003 / −0.020, `V_rot` +0.040 / +0.013 / +0.001 /
−0.120 (the s123 `np` run early-stopped at epoch 3 on val hidden-MSE, the same selection trap as
§8ay's inclusive arms; its `V_rot` skill vs PCHIP is +0.009 against the control's +0.128). `tm_pre` vs `tm_scratch`, the one controlled pair inside the card: `T_i` −0.094 vs
−0.090, `V_rot` +0.012 vs +0.021 — the MAST-pretrained backbone is indistinguishable from a random
initialization of the same architecture on every cell.

### 2. Axis 3 — where the arms win (cut population)

**Gap length** (hidden-run length the point sits in), `V_rot`, skill vs PCHIP:

| stratum | n | `bilstm` | `np` | `tm_scratch` | GP |
|---|---:|---|---|---|---|
| 1 step | 1,236 | −0.258 [−1.22, +0.10] | −0.233 | −0.253 | +0.012 |
| 2–10 steps | 3,260 | **+0.629** [+0.07, +0.85] | +0.646 | +0.633 | +0.082 |
| > 10 steps | 13,801 | **+0.386** [+0.04, +0.63] | +0.409 | +0.435 | +0.016 |

Inclusive, same strata, `bilstm`: −0.246 / +0.441 / +0.439 (GP +0.063 / +0.046 / +0.048); the arms tie
the control in every inclusive stratum except `tm_pre` > 10 steps (−0.043, WORSE).

Every learned arm loses to PCHIP in one-step gaps and beats it by +0.4–0.6 once the gap is longer
than the anchor's reach — the same shape §8al measured causally (saturation at 50 ms). Offline,
with the diagnostics inside the gap, `V_rot` *is* recoverable from the fast channels at the
100 ms scale; it is the 10 ms scale, where the adjacent CES sample is a better estimator than
anything the diagnostics say, that §8ab's `V_rot` n.s. lives in. The arms do not differ from
each other in any stratum (δ within ±0.05, CIs include 0 except `tm_scratch` > 10 steps +0.049
[+0.001, +0.079]).

> **Correction (2026-10-13, §8bb P02).** "Recoverable from the fast channels" cannot be what the
> `bilstm` control shows: `b12_models.py::BiLSTMRouted` keeps seq_v2's routing, so its `V_rot`
> branch reads `x[..., N_FAST:]` and never sees BES/ECEI/MC. The +0.39…+0.63 in the > 1-step
> strata is therefore the **two-sided label context** (the learned bridge, `Var_bridge(g_p, g_f)`
> of P02) and not diagnostic information; the only arm whose `V_rot` path sees the fast channels is
> the transformer, and its excess over the routed control, +0.049 [+0.001, +0.079], is the upper
> bound on what the diagnostics add inside a long gap. Consistent with P02's raw-CSV test: the
> `V_rot` increment is uncorrelated with BES/ECEI/MC activity at every window from 10 to 200 ms
> (|r| ≤ 0.014) while `T_i` shows +0.15…+0.28. Read "offline gap-filling recovers `V_rot` at the
> 100 ms scale" as a statement about having a future anchor, not about the diagnostics.

**Label-sparsified sets** (dense test shots keeping k `V_rot` labels; the H-np test bed, §3):

| k | `bilstm` | `np` | `tm_pre` | `tm_scratch` | GP | δ `np` vs control |
|---:|---|---|---|---|---|---|
| 0 (ref = train mean) | −0.025 [−0.06, +0.01] | −0.040 [−0.07, −0.02] | −0.016 | −0.003 | 0 | −0.015 [−0.055, +0.022] TIE |
| 3 | +0.146 | +0.148 | +0.132 | +0.137 | +0.127 | +0.002 TIE |
| 10 | +0.407 | +0.403 | +0.392 | +0.398 | +0.079 | −0.003 TIE |

With **zero** same-shot labels no arm beats the training mean — the `np` arm, whose whole
hypothesis is a shot latent that carries the regime, is the worst of the four there (−0.040,
CI excludes 0). With 3 labels all arms sit on the GP (+0.13–0.15); with 10, all arms are +0.40
and the GP +0.08. The gain from 3 → 10 labels is the same for every family: it is the context,
not the encoder, that carries the shot. The inclusive population repeats all nine cells (every
`np` δ within ±0.004, TIE; `tm_pre` WORSE at k = 3 and 10).

### 3. Axis 6 — cost

| arm | params | GPU minutes / run (cut, 4 splits) | best epoch |
|---|---:|---|---|
| `bilstm` | 255,938 | 7–14 | 12–32 |
| `np` | 309,794 | 3–20 | 3–36 |
| `tm_pre` / `tm_scratch` | 1,823,490 | 1.5–6 | 5–39 |

### 4. Verdicts (pre-registered §6, unchanged)

- **`V_rot` (the first question): no arm improves on the control.** In the cut population `np`
  and `tm_scratch` pass the §6 rule by a CI low of +0.000 and +0.001 (δ +0.021 each; the `np` cell
  read INCONCLUSIVE at a different bootstrap seed, which is what a CI low of 0.000 means), `tm_pre`
  ties, and no stratum reproduces either cell. In the inclusive population all three arms are
  TIE (δ −0.009 / −0.016 / −0.003), so neither cell is unconditional. Recorded as **TIE for the
  family**, with the two cut-population edge cells named.
- **H-np is not supported**: the shot latent buys nothing in the label-poor regime it was built
  for (k = 0 TIE and below the mean; k = 3, 10 TIE), and nothing in the real 1–20-label stratum
  (δ +0.001 on 81 points). §8's second pre-named reading applies: the 28 % between-shot `V_rot`
  variance is set by operating variables the diagnostics do not encode. The named next
  measurement is `np` conditioned on shot metadata (beam sources, `I_p`, `B_T`).
- **H-tm is not supported, in the cleanest possible form**: pretrained and random backbones agree
  to 0.004 / 0.007 on `T_i` and 0.009 / 0.013 on `V_rot` (cut / inclusive). Cross-device pretraining on MAST transfers no CES
  gap-filling knowledge to KSTAR — the same sign and size as TokaMind's own profile-group result
  (−0.005). The transformer family is also *worse* than the BiLSTM on `T_i` (cut −0.09, inclusive
  −0.03 to −0.04, WORSE in both) at 7× the parameters, consistent with §8ag's operator-family tie
  at this data scale.
- **`T_i`, not asked but measured**: `np` is +0.026 [+0.004, +0.049] over the control, signs 3/4,
  IMPROVE by the rule in the cut population, and a TIE (−0.006) in the inclusive one — cut-
  conditional, like B.14's rung. It is the same size as the B.15 ensemble's +0.02 and sits at the
  tie threshold; it is reported as a real but marginal effect of cross-attention over the
  same-shot context, not as a family win.

### 5. What this settles, and what it names

1. The model-family axis closes the way the estimator axis did (§8ag, §8aa, §8e): three
   inductive biases, one tie on the primary target. The diffusion / MAE / cross-modal arms keep
   their pre-registered hypotheses (distributional axis; label-count bottleneck; modality
   contribution) but the point-skill expectation for them is now set by this card: tie.
2. The k = 0 row is the sharpest statement of §8ar yet: with no same-shot label, *no* learned
   arm recovers `V_rot` above the training mean from BES + ECEI + MC alone. The information is
   not in the inputs; §8's first pre-named path — NBI torque, µs Mirnov, re-analysed labels — is
   the only one left, and the k = 3 → 10 curve (+0.15 → +0.40 for every arm) prices the third.
3. The gap-length profile is a positive result for the offline product: at the 100 ms scale the
   diagnostics recover `V_rot` at +0.4–0.6 over interpolation, which the causal 10 ms setting
   cannot show. The offline gap-filler and the causal nowcaster answer different questions and
   the paper should say so with this table.

Artifacts: `data/.b12_card.json`, `data/.b12_{bilstm,np,tm_pre,tm_scratch}_cut{3000,0}_s{42,1,7,123}/`,
`data/.b12_batch.log`, `data/.b12_runner_stdout.log`.

---

## 8ba. The derivation audit (2026-10-13) — every design choice and headline derived from first principles, and each derivation checked against the frozen record

**Question (승상님).** "우리의 연구를 뒷받침할 물리학적·수학적 근거 전부 유도" — assume both are
derivable, and keep going until each is proven. §8ar put the two conservation laws and three time
scales beside the results; this closes the rest: the measurement physics of CES, the estimation
theory the framing rests on, and the statistics of every evaluation device.

**Design** (`docs/theory_foundations.md`, 17 sections; `ces_prediction/analyze_foundations.py` →
`data/.foundations.json`, log `data/.foundations.log`). No model trained, scored, selected or
re-selected. Data: the 641-file census under the confirmed treatment (held-free, 3 keV cut) and the
frozen TEST npz of `.b1_seqv2_s42_i{42,1,7,123}`, `.b5i_seqv2_s42`, `.b15_ens4_cut_s42`. Each
derivation ends with one of three kinds of check — an **identity** that must hold on the frozen
artifacts, a **prediction** compared with an independently measured number, or a **Monte Carlo**
against the closed form.

### 1. What the derivations predict, and what the record says

| derivation | predicted | measured | status |
|---|---|---|---|
| CRB of a Gaussian line fit: `σ_v = v_th/√N`, `σ_T/T = √(2/N)` — the two noise floors are one photon budget | `N_T = 326`, `N_V = 258` | MAD 46.4 eV / 4.3 km/s (§8aq) | same order, ratio 1.27 (≤ 2 with finite-N width inefficiency) |
| Fisher `I_μσ = 0` for a symmetric line → shift and width estimates orthogonal | corr 0 | MC +0.04; masks agree 72.8 % (§8at) | consistent with independent missingness |
| `τ_eq = (m_i/2m_e) τ_e` is the Spitzer result, not a convention | NRL exact / convention | ratio 1.03 everywhere | confirmed |
| first-order relaxation driven by `T_e` ⇒ corr(level) > corr(rate) > 0 | 0.42 > 0.02 (sim) | 0.31 > 0.078 (§8ar) | ordering confirmed |
| NBI torque `T/P = 2 R_tan / v_b ∝ R_tan E_b^{-1/2}`; radial force balance terms 12 vs 30 kV/m | — | `T_e`~`V_rot` null (§8b.3) | power ≠ torque, derived |
| decimated kHz mode: `r_1 = cos(2πf_0Δ)·exp[−(σ_f/22.5 Hz)²]` | ≈ 0 for σ_f ≥ 50 Hz | −0.009 (§8b.2); MC −0.03…+0.03 | confirmed |
| one decimated sample carries amplitude at relative error `1/√2` | 0.707 | MC 0.70–0.72 | confirmed; data-processing inequality closes derived MC features |
| noisy OU fitted to the 641 files (robust `T_i`: σ_n 39 eV, σ_s 69 eV, τ 81 ms): one-step RMSE ratio vs persistence | interp 0.81, filter 0.88, last-1 0.91 | median ratios s42: PCHIP 0.85, linear 0.83, GP 0.84, causal GP 0.96, **model 1.03** | within 10 %; the MSE backbone carries its anchor, bound 0.875 |
| Markov + noise: value of history beyond one observation | K=2 +6.8 %, K≥5 +8.1 % (saturates) | W = 2 plateau (§8f), reach deficit −0.065 @ 20 ms (§8af), saturation 50 ms (§8al) | consistent; 5 steps = 50 ms |
| filter memory `1/−ln(ρ(1−K))` | 1.2–1.8 steps | 50 ms ≈ 3–4 memory lengths | consistent |
| history-only causal vs two-sided interpolation by gap | −0.21 (10 ms) … −0.80 (200 ms) | causal GP median ratio 1.12–1.25 at every gap; model +0.17…+0.39 MSE | sign confirmed, monotone growth unresolved (120–600 rows/bin, no future-anchor key) |
| causal skill vs persistence = noise averaging + `Δ/2τ` dynamics | 0.331 (classical fit), dynamics ≤ 0.067 | causal GP +0.310, model +0.358 (s42, Δt ≤ 15 ms) | within 0.02; fast diagnostics add ≈ +0.05 at one step |
| PCHIP amplifies anchor noise through its slope terms | 1.027 (MC on fitted OU) | 1.022 (`T_i`), 1.043 (`V_rot`) | confirmed |
| spike dilution `skill' = 1 − (1−f_m)SSE_m / ((1−f_b)SSE_b)` | +0.589 | +0.225 → +0.589 after the 3 keV cut (incl s42, 110 rows) | identity; spikes carry 54 % of PCHIP SSE, 75 % of the model's |
| MSE = ∫Q²dp ⇒ skill is quantile-weighted | running contribution | top 1 % of rows (by `se_m+se_b`) carry 62 % of `T_i` skill, 100 % of `V_rot`'s; q75 crossover | identity; the §8ax tail reading, as algebra |
| cluster design effect `1 + (m−1)ρ` | — | 3.1 (`T_i`), 1.4 (`V_rot`): naive CI 1.8× / 1.2× too narrow | measured |
| bounded loss sees only the bulk share of val MSE | 10 % (cut), 5.9 % (incl) | epoch 2–4 early stop (§8ay) | mechanism quantified |
| ensemble gain = mean member disagreement variance / MSE_b (exact) | +0.0097 | +0.0090 vs canonical (B.15 s42) | identity; Jensen holds on 100 % of rows; the +0.02 tie is structural |

### 2. Verdict

1. **Nothing in the record is contradicted by a derivation**, and three results that were empirical
   get a mechanism with a number: the 50 ms saturation (two independent scales now — `τ_eq` 8–59 ms
   and the noisy-Markov memory of 1–2 steps saturating at 5), the W = 2 plateau (Markov property; the
   residual +8 % of MSE is what the paired reach ladder saw and the unpaired sweep could not), and the
   bulk ordering GP ≈ PCHIP < causal GP < persistence ≈ backbone (noise-averaging arithmetic, 7.3).
2. **Two numbers the derivations add that the record did not have.** The deployable causal bound in
   the bulk is 0.875× persistence's median error; the MSE backbone sits at 1.03× and the Huber rung at
   0.94× — so B.14 recovered half the headroom the OU model says exists. And at one-step gaps the fast
   diagnostics are worth ≈ +0.05 of MSE skill over the best history-only filter: small in aggregate,
   which is the §8ax tail concentration seen from the theory side.
3. **The causal GP is not the optimal causal baseline.** It lands at 0.96× persistence in the median
   where the one-observation optimal shrinkage alone gives 0.91× and the full filter 0.875×. A
   Kalman/OU arm is a better deployable control than the 16-neighbour local GP, and is named below.

### 3. What this does not show, and the measurements it names (§8j rule)

- The OU fit is a Gaussian, stationary, single-scale model of a target whose aggregate is tail-shaped;
  its bulk (robust) and aggregate (classical) fits differ by 3.6× in σ_n and are used only for the
  statistic each describes. The plasma parameters behind `τ_eq`, the NBI geometry and the pedestal
  force-balance magnitudes are quoted assumptions (appendix A of the document).
- Named next measurements, none of which retrain anything: (i) score an exact OU/Kalman causal arm
  on the frozen rows so `skill_model − skill_filter` isolates the fast-diagnostic information per gap
  and stratum; (ii) add a future-anchor-distance key to the npz (additive, §5.8-C5) to resolve the
  bridge-vs-extrapolation curve of 7.7; (iii) an anchor-denoising input arm against the 0.875 bound;
  (iv) when CES per-frame fit quality arrives, read `N` directly and replace the value cut.

Artifacts: `docs/theory_foundations.md`, `ces_prediction/analyze_foundations.py`,
`data/.foundations.json`, `data/.foundations.log`.

### Addendum (same day) — the plasma-behaviour physics, and the two routes that predict CES without CES

**Question (승상님).** "CES 자체를 예상하는 물리는 설계된 건 없어?" — not the logic by which CES arises, the
behaviour physics of the plasma. `docs/theory_plasma_physics.md` (12 chapters: single-particle motion,
equilibrium, kinetic → two-fluid, collisional and neoclassical transport, turbulence and stiffness, H-mode /
pedestal / ELM / RMP suppression, rotation, ion power balance, MHD modes, sources, global scaling,
synthesis) and `analyze_foundations.py` **Part F**.

**The two routes.** `T_i` is the low-pass of `(T_e, n_e, P_NBI,i)` through the ion power balance with
`τ_eq ∥ τ_E,i`; `V_rot` is the low-pass of torque through `τ_φ ≈ τ_E`, **and an MHD mode's lab-frame
frequency is a Doppler measurement of it**: `v_φ = 2πR f / n`. The dataset's EHO harmonics (~4 / ~8 kHz,
§8ao) read as n = 1 / n = 2 at R = 2.1 m give **53 km/s** against the CES `V_rot` median of 69 km/s —
self-consistent harmonics and the same order. That route lives only in the kHz Mirnov stream (B.6).

**Measured on the 641 files (per-block 2SLS with `y_{t−1}` as the instrument, held-free, 3 keV cut).**

| fit | blocks | τ_eff median (IQR) | note |
|---|---:|---:|---|
| `T_i`, no driver | 598 | **140 ms** (52–307) | OLS gives 36 ms: the noise-attenuation bias foundations §7 predicts |
| `T_i` + ECEI driver | 598 | 103 ms (44–263) | driver sign positive in **46 %** of blocks — not identified |
| `V_rot`, no driver | 342 | **89 ms** (38–252) | OLS 32 ms; held removal biases low |
| IPB98(y,2) τ_E, KSTAR-like | — | 77 ms | assumed I_p 0.6 MA, B 1.8 T, P 3 MW, n 3e19, R 1.8, a 0.5, κ 1.8 |

So the energy and momentum relaxation times of the CES series are both ~100 ms, equal to the global
confinement scaling and to the OU τ of 81 ms fitted in Part D — **an effective Prandtl number of
0.6–1.4 read from the targets' own dynamics.** Other numbers: pedestal ρ* 0.0055, ν*_i 0.24 (banana),
β 0.45 %, χ_neo 1.0 m²/s, χ_gB 0.12, E×B shearing 8×10⁵ s⁻¹ vs ITG growth 7×10⁵ (H-mode condition met at
the assumed E_r), NBI critical energy 18.6 keV at T_e 1 keV so a 100 keV beam heats electrons 71 % (ion
heating arrives through `Q_ei`, which is why `T_i` follows `T_e`).

**Verdict.** The physics that predicts the targets exists and is written down as two boxed generative
models (document §12). What the data confirms: the relaxation times and Pr ~ 1, the Doppler reading of
the EHO, the OLS bias. What it refuses: the e–i coupling term cannot be identified from the uncalibrated
four-channel ECEI mean (sign random across blocks), so the physics-based `T_i` predictor needs one
calibrated `T_e` at the CES flux surface. Named measurements, in the order of the equations they close:
calibrated `T_e(ψ_CES, t)`; kHz Mirnov mode frequency / mode number (→ `v_φ`) and band power (→ NTV);
NBI and RMP coil waveforms (→ torque); a `D_α` / ELM indicator (→ the impulse term).

---

## 8bb. Ten parallel proofs: each headline result derived from the plasma's time-domain equations and checked against the record (2026-10-13)

**Question (승상님).** "많은 sub들의 증명을 통해서 우리의 결과의 관련성을 어떻게 설명할지 구상하고 이를
물리로 표현 — 특히 t에 관한 수식으로." Ten independent proof agents each took one result of the
thesis, derived it from first principles as explicit equations in time t, and checked the
derivation on the 641-file census and/or the frozen TEST artifacts. The synthesis is
`docs/theory_results_physics.md`: chapter 0 states one time-domain generative model (M1–M7) that all
ten results fall out of; chapter 1 holds the ten proofs verbatim; chapter 2 the relevance narrative
and the measurements the proofs name. No model was trained, scored or selected. Scripts and their JSON outputs live in
`ces_prediction/experiments/proofs/p01`–`p10`; every number below is transcribed into the document.

### 1. The one model

`dy/dt = −(y − m)/τ + sqrt(2σ_s²/τ) ξ(t) + Σ_k A_k ḣ(t − t_k)`, `h(s) = Θ(s) e^{−s/τ_rec}`,
observed at `t_j = jΔ` with photon noise `n_j` and an acceptance indicator `m_j = 1[N(t_j) > N_c,
B(t_j) < B_c]`. The two targets share the form; what differs is where the coefficients come
from — `T_i`'s equilibrium and impulses are set by `(T_e, n_e, P_NBI,i)` that ECEI/BES observe at
the same instant, `V_rot`'s by torque that nothing in the dataset observes. The paired difference
of any two estimators integrates to `D = p_obs · r · N_e κ A²/n − c_b` (P08) and the MSE skill is
exactly `skill = f_I r_I − (1 − f_I) ℓ_B` (P01).

### 2. What each proof established

| proof | result explained | derivation → number | check |
|---|---|---|---|
| P01 `T_i` headline is transient physics | +0.24 vs PCHIP 4/4 (§8ab) | identity `skill = f_I r_I − (1−f_I) ℓ_B`; the transient term alone gives **90–114 %** of the headline on 4/4 splits (+0.199/+0.223/+0.244/+0.265 vs +0.174/+0.248/+0.257/+0.264) | quiet half: model loses to the acausal GP 4/4; top-5 % rows: beats the GP 4/4 (+0.05…+0.09); `ΔT_i`–`ΔECEI` corr 0.142 → **0.238** conditional on top-5 % jumps, lag 0 only |
| P02 `V_rot` tie from momentum balance | 1/4 vs PCHIP, 3/4 vs pers, median −49 % (§8ab, §8ax) | unobserved torque ⇒ Bayes-optimal causal = history-only OU filter; median bound 0.921× persistence; aggregate > 0 iff tail gain `G_T` > bulk deficit `D_B` | s42: model median 1.14×, `G_T − D_B = 0.414 − 0.024 = 0.390` = skill; top-1 % (105 rows) carries all of it; `ΔV`–`ΔBES/ECEI/MC` = 0 at every lag and 10–200 ms window (`T_i` control +0.15…+0.28); **§8ab block 2's 43 % PR2 fallback is a counting artifact — 0.3–0.5 % of scored rows** (corrected in place) |
| P03 50 ms saturation | §8al | cold-start Kalman from `P_0 = σ_s²` reproduces `MSE_K` exactly and saturates (2 %) at **2–3 steps, not 5**; the measured per-step decay 0.559 (memory 3.4 steps) would need K = 0.155 | the OU cold start explains **26–43 % of the 20 ms deficit** (RMS vs the 8al curve 0.021–0.026, curve RMS 0.032) — rejected as the main cause; the rest sits with the ELM-recovery scale: 16,703 ≥ 25 % one-step drops (7.7/s, 26 % single-point dips), cleanest recovering class n = 2,783 with **τ_rec = 60 ms** (55–64 ms across definitions), `τ_eq` 18–39 ms; a 3-parameter recovery fit clears the noise floor only at `t_ctx ≈ 1.5 τ_rec`; 50 ms is defended as order-of-magnitude only |
| P04 campaign drift is instrument physics | §8n, §8s | `x_f(t) = g(D_f) s_f(t) + c(D_f)`; within-shot z-score cancels `g, c` exactly; EWMA bias on a trend = `−b τ_w` | between-shot variance share BES 0.84 / ECEI 0.91 vs `T_i` 0.49 / `V_rot` 0.28; §8n's 1.218/0.529/0.115 σ reproduced to 3 decimals; ECEI mean ratio ≈ std ratio (multiplicative gain); absolute level adds 1.4–2.6 % R² to the per-shot gain; EWMA τ_w = 1 s measured 0.29–0.33 σ vs derived `b τ_w` 0.31/0.27 |
| P05 3 keV cut is a pressure limit | §8ab spike audit, §8q | [P05] | [P05] |
| P06 fast-diagnostic information bound | ≈ +0.05 at one step (§8ba) | `Δskill = r²_{yz·h} · MSE_h/MSE_pers`; the right conditioning variable is the filter innovation, and the channel is denoising of the noisy anchor, not the rate | history-conditional linear bound **+0.047–0.056** vs measured **+0.048**; implied partial r 0.26, I = 0.036 nat; §8ar's rate correlation (0.07) gives +0.003 — not the right input; MC = 0 in every conditioning; 81–83 % of the gain in the top-10 % innovation rows; per-step information `I_step = ½ ln(1 + tanh(Δ/2τ_eq) S_z)` compatible with the measurement only for τ_eq ≲ 30–60 ms |
| P07 MNAR is photon-budget physics | §8i, §8ab 9, §8at | `ln N(t) = … − σ_s n̄_e(t) L`, `B(t) = B_0 + B_1 e^{−(t−t_0)/τ_B}`, CRB with background `σ² ∝ (1/N)(1 + c B/N)`, width 2–3× more background-sensitive than centre | missing rate vs neighbour-jump quintile 0.77 → 4.45 % (×5.8, both targets); BES lowest quintile 19.9 % vs top 3.7 %; `τ_B` ≈ 25–35 ms measured after BES drops; mask agreement 72.77 % (§8at 72.8), φ = 0.041; ECEI increments do **not** predict missingness (recorded as a failed prediction) |
| P08 win-rate asymmetry | 70 % vs 48 %, terciles 85 vs 55 % (§8al, §8an) | `D = p_obs r E_imp − c_b`; `κ_T ≈ 1.03` (step) vs `κ_V = Δ/2τ_φ = 0.056` (ramp) | terciles 43/85/86 vs 36/45/59 reproduced; quiet-tercile loss −0.072 = −0.073 (= `−c_b`, driver-independent as predicted); impulse rows win 0.744 (`T_i`) vs lose 0.425 (`V_rot`); `p_obs` proxy enrichment 1.27–1.34× for `T_i`, 0.99× for `V_rot`; logit `p_obs` +1.92 ± 0.53 (`T_i`) vs −0.55 ± 0.44 |
| P09 MAST transfer null by similarity | §8az | only the form of (M1) is device-independent; all 7 coefficients device/instrument-specific; `SE(ρ̂) = sqrt((1−ρ²)Δ/T)`, 10 % precision at `T ≈ 200 τ` | β 7×, ε 2.75×, ρ* 1.6–2×; pretrained weights = 98.2 % of 1,823,490 and explain a one-line equation; one median shot (3.98 s) pins ρ to 2.4 SE of the MAST prior's bias; k 3→10 lift spread 0.006 across four families |
| P10 real-time classes from plasma time scales | §8ac, §8aj, §8l | `MSE(ℓ) − MSE(0) ≈ (ℓ/Δ)[MSE(Δ) − MSE(0)]`; three latency classes A/B/C | latency cost −0.023/ms (robust) / −0.009/ms (classical) in skill, 1.5 eV at 1.5 ms; nowcaster (p99 1.5–9 ms) is class A only, classes B/C excluded by the 10 ms label integration, not by inference; 353,920 MAC/step reproduces §8ac; 39-op theoretical floor vs 111 shipped; recovery-type events τ_rec 25–37 ms |

### 3. Verdict

1. **Every headline number has a physical cause written as an equation in t**, and the causes are
   the two sides of one fact: the energy equation's right-hand side is observed by the fast
   diagnostics at the same instant, the momentum equation's is not.
2. **Two corrections to the record.** The §8ab "43 % PR2 fallback" is a counting artifact (P02;
   annotated in place). §8ar's rate correlation (+0.07) is not the measure of the `T_i` coupling
   the nowcaster uses; the history-conditional partial correlation (0.26) is (P06).
3. **Named measurements** (each proof's §4): `D_α`/ELM indicator (P01, P07, P08); kHz Mirnov mode
   frequency as the `V_rot` input and the test that `V_rot` impulse-row win rate inverts (P08, P02);
   per-frame CES fit quality to replace the value cut (P05, P07); a calibrated `T_e` at the CES flux
   surface (P06, P03); an exact OU/Kalman causal arm scored on the frozen rows (P02, P06); a
   same-estimator online standardization arm (P04).

Artifacts: `docs/theory_results_physics.md`; `ces_prediction/experiments/proofs/p01`–`p10` (scripts + JSON).

## 8bc. B.16 registered: the physics-structured nowcaster (2026-10-13) — built and unit-tested, not run

**Question (승상님).** "구한 물리적 지식을 기반으로 더 나은 모델을 설계하고 만들어줄 수 있어?" The
derivation audit (§8ba, §8bb) located two things the MSE backbone does wrong and where: in the
quiet bulk it carries the noisy anchor (median 1.03× persistence) while the noisy-OU optimal
causal filter sits at 0.875× (foundations §7.4; Huber reached 0.94×, §8ay); and its gain over
interpolation lives only in rows where a pedestal event is visible in the fast channels at the
same instant (P01: the transient term is 90–114 % of the headline; P08: quiet discharges lose by
`−c_b`). Both point at the **output structure**, not at capacity or the loss.

**Design** (`experiments/PREREGISTRATION_B16.md`, committed before any number;
`experiments/seq/model_seq_v4k.py`, registry names `v4k` / `v4kng`; runner
`experiments/b16_physics/run_b16.py`). The output is `ŷ(t) = a(t) + g(t)·r(t)`:

- `a(t)` — a **learnable scalar OU–Kalman cell** per target in normalized units (state `x, P, m`:
  update with the one-step-lagged observation read from the carried feature and the shifted loss
  mask, then propagate `x ← m + ρ(x−m)`, `P ← ρ²P + σ_s²(1−ρ²)`; `m` is a slowly adapting shot
  level). Parameters `(ρ, σ_s², σ_n², α)` are learnable and initialised from the §8 OU fits; the
  learned values are written back in physical units (`physics_params.json`) as an interpretability
  check (H16-f).
- `r(t)` — the seq_v2 LSTM residual, with the anchor and its predictive sd appended to the LSTM
  inputs. Head zero-initialised → **training starts exactly at the Kalman filter** (B.3's
  persistence start, one rung up).
- `g(t)` — an event gate: for `T_i` from the LSTM state, the fast-channel increment energy
  `log1p(mean Δz²)` (the P08 `p_obs` proxy) and the anchor sd; for `V_rot` from the slow state
  only. Bias −2 at init. `v4kng` pins `g ≡ 1` and isolates what the gate buys.
- Routing is seq_v2's at the encoder; nothing on the `V_rot` path reads the 15 fast channels.

**Unit tests** (`tests/test_seq_v4k.py`, 6 pass, CPU seconds): registry and shapes; `V_rot` output
bit-identical under fast-channel perturbation while `T_i` responds; causality (rows > t do not
change the prediction at ≤ t); the model at init equals its Kalman anchor; the cell's predictive
variance converges to the Riccati fixed point of foundations §7.4 and decays by `ρ` per step with
no observations; the physical readout recovers the initial `τ, σ_n, σ_s`.

**Pre-registered predictions** (§3 of the document): cut `T_i` median 0.875–0.94× persistence
(M1 vs backbone +0.05…+0.12); quiet-tercile discharge win rate vs the causal GP ≥ 0.50 for `v4k`
(backbone 0.43); aggregate MSE skill vs PCHIP a tie (±0.02); cut `V_rot` median 0.92–1.01×;
learned `τ` 40–300 ms. Rules H16-a…f and the adoption / replacement thresholds are fixed in §4.

**Status.** 승상님 (2026-10-13): no GPU batch today, documents only; the batch runs the next day
(`run_b16.py --smoke`, then 16 runs ≈ 30–50 min on the RTX 5060, then
`rescore_b13.py --extra v4k=.b16_v4k_{pop}_s{s} v4kng=.b16_v4kng_{pop}_s{s}`). Nothing here
touches TEST; no number exists.

## 8bd. B.6 executed (2026-10-26): the raw microsecond stream arrives, the Mirnov physics is real, and the restored features make `V_rot` worse — H1 rejected in both populations

**Question.** `PREREGISTRATION_B6.md` (frozen 2026-08-21, §8ao) asked one confirmatory question:
does restoring the alias-destroyed kHz Mirnov information — causal 10 ms window features from the
2 MHz coil stream, fed to the `V_rot` branch only — improve `V_rot` nowcasting over the same model
without them (H1)? Everything else in the batch is measurement: the §1.4 positive controls, the
H3 companion-leakage pairs, the §4.5 time-ordered re-split and the H4 transition figures.

**Delivery (the ⟦on-arrival⟧ slots, filled in commit `7d42c48` before any TEST number).** 승상님
placed 36 CSV files (12 frozen discharges × BES / ECEI / MC, 7.4 GB) in the thesis `data/` folder;
they now live under `data/raw_mus/` (gitignored) with an npz cache and `delivery_manifest.json`
written by `b6_mus/raw_loader.py`. No metadata came with the files, so the sampling rates are
**measured** from row count and span and verified sample by sample against the delivered `time`
column (36/36 uniform clocks): **BES 500 kHz (#31747 alone 1 MHz), ECEI 500 kHz, MC 2 MHz** — the
PanoMHD-quoted 2 MHz for Mirnov, exactly as PROJECT_KNOWLEDGE expected. The `time` column is
quantised (1 µs; 10 µs for #31873 and #31921), so timestamps repeat and the loader returns the
reconstructed uniform clock. Two coverage defects: **#31873's MC file holds 2.0 s of the 7.67 s
requested** (5.41–7.41 s; BES / ECEI are complete), and #31902's raw window starts at 1.75 s so a
third of its labelled `T_i` rows fall outside it (every `V_rot` row is inside). Zero NaN.

**§1.4 positive controls** (`raw_positive_controls.py` → `data/.b6_positive_controls.json`; the
criteria are the registered ones, the channel names and windows are slot fills):

| # | control | result |
|---|---|---|
| 5 | #32092 Mirnov EHO 4 / 8 kHz, 7.3–7.9 s | **PASS** — Welch peaks at **4.39 kHz** (prominence 6.8 × 10⁴ on both coils) and **8.91 kHz** (60× / 213×); the extractor's own mode channel reads the 60 rows of that window as 100 % valid, modal frequency 4 kHz, median 4.8 kHz |
| 3 | #31921 MC envelope at 7.3 s | **PASS** — RMS ratio 7.88 (≥ 3) |
| 4 | every channel, snapshot² vs windowed raw mean square | **176 / 180 in band**; the four misses are MC coils (0.36–0.47) on bursty, kurtosis-12-to-17 signals where a few hundred windows cannot pin a second moment (all-row ratios 0.29–1.14, bootstrap CIs ±0.2–0.5). The stronger identity holds everywhere: **on 24 / 24 coils the 100 Hz MC value equals a raw sample at the grid time** (lag 0, error 0 or within the print quantum) — the §8b.3 "random-phase snapshot" premise is now a measured fact, not an inference. No acquisition fault; the four coils stay in the arm (a sensitivity row without #31902 is below) |
| 1 | #31923 `BES_0206` 3–6 s, 30–70 kHz WCM | **FAIL at the registered window** — band ratio 2.07 (< 3). In 3–6 s the BES spectrum is a white floor above 10 kHz and adjacent-channel coherence is ≈ 0. After the published L → FIRE transition at 6.1 s a broad coherent band appears at 60–130 kHz (coherence 0.6–0.75, three channel pairs) with partial 30–70 kHz coherence (0.3–0.5); the MC spectrogram shows a 30 → 45 kHz chirping mode and a 10 kHz line in L-mode. The registered "3–6 s" was transcribed from the 100 Hz file's quiet stretch (`SELECTION.md`), not read off the paper's figure, and the paper is behind the IOP wall — so the control is recorded as failed, not re-windowed |
| 2 | #32027 BES pair, 100–300 kHz cross-power | **FAIL** — delivered BES has only 0105–0307, so the poloidal pair (0106, 0206) around the literature channel was used; windows with a valid coherent peak in 100–250 kHz = 1.8 % (slot threshold 5 %). Long-window coherence is ≈ 0 above 100 kHz throughout (10–100 kHz turbulence is alive). The BES 70–300 kHz band is unusable on these channels |

By the registered rule a failed control excludes its channel: BES is excluded, and BES was never in
any arm (§3.2 was exploratory and is not extracted). The MC confirmatory path is clear.

**Features** (`mus_features.py` → `data/.b6_features/`, K = 15, all five bands under the 1 MHz
Nyquist): per 10 ms grid row, RMS envelope × 2, log1p band power in 0.1–3 / 3–10 / 10–30 / 30–70 /
70–300 kHz × 2, and the two-coil cross-spectrum's dominant peak (log1p frequency, cross-phase sign,
validity under the coherence floor). Validity is 95–99.7 % of covered rows in every discharge;
median dominant frequencies run 6 kHz (#32092, the EHO) to 40 kHz.

**Design** (unchanged from the pre-registration, `run_b6.py`): A0 = seq_v2 on the six pool
discharges with the 100 Hz inputs; A1 = A0 + the 15 features into the `V_rot` branch; A2 = A1 minus
the two decimated MC channels. For every (arm × population): six leave-one-out folds (init 42,
cap 100, patience 6), median best epoch, four refits (inits 42 / 1 / 7 / 123) at that epoch on all
six, TEST scored once. Fold best epochs 4–46, cap never binding; refit epochs 14 / 18 / 16 (cut) and
19 / 18 / 19 (inclusive). 66 GPU runs, ~2 h on the RTX 5060 (every training subprocess exits with
the known Windows CUDA-teardown code after saving; the runner accepts artifacts on disk).

**Result — H1 (`V_rot`, A1 − A0 paired on identical TEST rows, n = 917 × 4 seeds; `verdict_b6.py`
→ `data/.b6_verdict.json`).**

| population | pooled skill (shot-clustered CI) | per-seed | test discharges improved | −top1 | block CI | verdict |
|---|---|---|---|---|---|---|
| cut 3 keV | **−0.128** [−0.544, −0.105] | −0.111* / −0.212* / −0.130* / −0.062* (4/4 negative) | **0 / 3** (31114 −0.11, 31902 −0.54, 31921 −0.11) | −0.136 | [−0.249, −0.033] | **REJECT** |
| inclusive | **−0.164** [−0.473, −0.133] | −0.174* / −0.192* / −0.192* / −0.099 (3/4 negative) | **0 / 3** (−0.13, −0.47, −0.20) | −0.234 | [−0.314, −0.076] | **REJECT** |

**H1: REJECT in both populations** — not undecided: the pooled CI excludes zero on the *wrong* side,
≥ 3/4 seeds are significantly negative, and the block bootstrap agrees. The features do not merely
fail to help; they cost a tenth to a sixth of the control's error budget. Sensitivity rows: A2
(decimated channels removed, so the model's only Mirnov view is the restored one) −0.172 / −0.156,
also 0 / 3; dropping #31902 (the coil that failed control 4) −0.110 / −0.150, 0 / 2 — the rejection
is not one discharge. H1-m is moot but reported: the loss concentrates in the middle and active
`V_rot` terciles (cut −0.46 / −0.12 vs quiet +0.04; inclusive all negative), i.e. the features hurt
exactly where the hypothesis said they would help. `V_rot` spike-row SSE share is 0 on this test set
(no |V_rot| > 1000 rows). `T_i` (descriptive, §5): −0.000 / +0.001, every CI covering zero — the
`T_i` branch is blind to the extras by construction and the batch confirms it.

**Context (reported, not judged).** A0 itself on these three `V_rot`-carrying test discharges, with
six training shots: `V_rot` vs PCHIP **+0.175** (cut) / **+0.206** (inclusive), 4/4 seeds positive;
A1 +0.070 / +0.076; A2 +0.034 / +0.083. The fold val MSE did not foresee the loss (A1 within ±0.07
of A0 on every fold, better on two) — six validation discharges of one shot each are too weak a
selection signal to catch a feature set that generalises worse than it fits.

**Why it hurts — the premise measured directly** (no model; `data/.b6_features` against genuine
`V_rot` rows of the same grid). Within-discharge Spearman correlation of the extracted **dominant
mode frequency** with CES `V_rot`: 31921 +0.08, 31114 −0.09, 31902 +0.05, 31747 +0.04, 32092
−0.11, 32097 −0.10 — six of seven discharges with ≥ 30 rows show |ρ| ≤ 0.11; only #31359 (ERMP
on, ETB) shows +0.51. The RMS envelope correlates weakly positively (+0.07 … +0.45) and the 3–10 kHz
band not at all. The physics chain behind H1 (modes co-rotate with the flow at their rational
surface → the Mirnov frequency is a Doppler readout of rotation, PROJECT_KNOWLEDGE "V_rot routing
chain") is correct about the *coil* — control 5 shows the EHO exactly where the paper put it — but
the coil's dominant frequency in a 10 ms window is not a row-by-row function of the CES edge
rotation in these discharges: it jumps between modes, carries the mode number as much as the flow,
and most of the time it is simply a different quantity. Fifteen such channels, per-shot z-scored,
on six training discharges, give the `V_rot` branch a high-variance input that fits the pool and
does not transfer; the result is the measured loss.

**Verdict.** The pre-registered falsification table (§8) applies verbatim: *"H1 미정/기각 → 100 Hz
파생 무효(기존, §8b.3 4-seed paired) + 원시도 무효 ⇒ Mirnov 경로 소진. '정보 부족'이 아니라 '이 채널에
없음'."* The Mirnov route to `V_rot` is closed at every rate on record — 100 Hz snapshots, 100 Hz
derived envelopes (§8b.3), and now the raw 2 MHz stream through the registered feature set.
**The named next measurement is unchanged and is now the only one left: the NBI torque channel**
(§8b.3 r = +0.024, §8an `T_e`-level null, §8az family tie). What this does **not** show: that no
function of the raw stream carries rotation (only the registered one was tested, as the rule
requires); a different feature family (e.g. the cross-phase *slope* across many coils, the
error-field paper's v⊥ method, which needs more than two coils) is a new pre-registration, not a
re-tune of this one.

### Addendum (same day) — H3 leakage, the §4.5 time-ordered re-split, and the H4 figures

**H3 — session leakage measured directly** (`run_b6_leak.py`, run after the verdict was locked,
outputs under `data/.b6_leak_*`, `data/.b6_leak.json`; 500 ms block bootstrap because each
measurement is one discharge). L1a adds companion #31923 to every A0 training set (folds and
refits) and reads the paired skill change on its test partner #31921; L1b adds #31357 to the
fold whose val shot is its partner #31359.

| | `T_i` cut | `T_i` incl | `V_rot` cut | `V_rot` incl |
|---|---|---|---|---|
| L1a 31923 → 31921 (n = 437 / 258 rows, 10 / 9 blocks) | **+0.154** [+0.026, +0.236] | +0.023 [−0.012, +0.054] | +0.059 [−0.067, +0.438] | +0.006 [+0.001, +0.089] |
| L1b 31357 → 31359 (n = 215 / 244, 6 / 5 blocks) | **+0.024** [+0.012, +0.086] | −0.012 [−0.077, +0.013] | −0.084 [−0.383, +0.053] | +0.109 [−0.037, +0.178] |

The same-session partner is worth **+0.15 of `T_i` skill on #31921 in the cut population** (ten
blocks, CI clear of zero) and +0.02 on #31359; in the inclusive population and for `V_rot` the
effect is inside the noise. Two cautions written with the number: L1a also changes the training
set from six to seven discharges, so the measured Δ is "one more shot that happens to be the
partner", the quantity §6 registered; and one L1a cut fold stopped at the 100-epoch cap (fold 2,
best epoch 100 — the B.3 lesson), so the cut refit epoch 29 is an upper-bounded median. Against the
100 Hz summary distance that put the two pairs at the 0.03rd percentile, the leakage is real and
`T_i`-sided — the companion rule in `folds.py` (never train on a partner across the test boundary)
is vindicated at the gate scale.

**§4.5 time-ordered re-split** (`run_b6_timesplit.py`, train = #31097 / #31359 / #31747, val =
#32027 / #32092 / #32097, `data/.b6_timesplit.json`): A1 − A0 on `V_rot` −0.025 (cut) / −0.102
(inclusive) with shot-clustered CIs spanning [−2.0, +1.0] — three validation discharges, one of
them with eight `V_rot` rows, decide nothing. Three training discharges do not learn `T_i` at all
(A0 vs PCHIP −0.61 / −1.17), so the row is reported as uninformative, as the pre-registration
allowed ("보고만").

**H4 — published transitions, three tiers** (`h4_figure.py`, `data/.b6_h4/h4_s{shot}_cut_s42.png`;
raw 2 MHz Mirnov spectrogram 0–50 kHz / `T_i` / `V_rot`, CES observations with the A0 and A1 cut
refits, never-trained discharges only). Qualitative, as registered:

- **#31921**: in L-mode the coil shows two stationary lines (~10 kHz and ~35 kHz) with the
  published FIRE marker (5.40 s) passing without a visible spectral change; at 7.3 s the broadband
  burst activity begins, `T_i` rises from 0.4 to 1–2 keV and both refits track it with the same
  shape; the H-mode marker (8.05 s) sits inside the bursty phase. A0 and A1 are nearly
  indistinguishable on `T_i` and differ on `V_rot` only in the bursty phase, where A1 runs a few
  km/s high.
- **#31923** (companion, `V_rot` held for the whole discharge): the L → FIRE transition at 6.1 s is
  the sharpest feature in the raw stream — the two L-mode lines end and burst activity starts
  within one window — and `T_i` jumps from 0.3 to 1.5 keV at the same instant, tracked by both
  refits. The `V_rot` panel shows the H1 mechanism in one picture: **A1's `V_rot` prediction spikes
  to 60–85 km/s on every Mirnov burst after 6.1 s while A0 stays at 10–40 km/s** — the restored
  features inject burst amplitude into the rotation estimate. The discharge has no genuine `V_rot`
  row to adjudicate, which is exactly why it is a companion and not a test shot.
- #31873 and #31357 figures are in the same folder (ELM-suppression trigger 4.5 s; ETB 5.5 s).

**H2** (attention vs recurrence on the non-uniform input) was not run in this batch; with H1
rejected its premise (a feature set worth re-encoding) is gone, and it stays open only as a
method question for a later session.

## 8be. B.6-T (2026-10-26): the raw burst energy that explains `T_i` increments inside a discharge does not transfer across six — exploratory, no gain, and the one artifact to fix first

**Question (승상님: "이 데이터로 최대한 가능성 있게").** `PREREGISTRATION_B6T.md`, written the same
day after two measurements pointed at the `T_i` side of the raw delivery. (i) The frozen B.1
backbone's TEST rows on the four raw test discharges, stratified by time since the last
raw-detected burst (`BES_0206` 1 ms energy, robust z ≥ 8): its gain over PCHIP sits in the
10–50 ms-after-burst stratum (+0.32 cut / +0.44 inclusive) while **rows with a burst inside the
causal 10 ms window show no gain (+0.01 / −0.13) and carry 51 % / 26 % of the model SSE**
(`data/.b6_burst_strata.json`). (ii) Within-discharge block-CV ceiling: adding raw 10 ms window
features (13 channels × {log1p max z, log1p mean z} + time since burst) to the 100 Hz snapshot
increments the backbone already sees raises `T_i`-increment skill vs persistence in **10 / 12
discharges** (median +0.04, best +0.25; `data/.b6_ti_raw_ceiling.json`). The same ceiling for
`V_rot` was empty (§8bd). So the registered arm T1 = A0 + 27 such channels into the `T_i` encoder
only (`CES_SEQ_EXTRA_TI`; routing unit-tested; frozen checkpoints reload unchanged), B.6's fold
protocol, A0's existing runs as the control, effect sizes only (`T_i` has no confirmatory rule at
twelve discharges, B.6 §1.3 / §5).

**Result** (`b6t_strata.py` → `data/.b6t_report.json`; refit epochs 15 cut / 11 inclusive vs A0's
14 / 19; the batch was interrupted once by the host's memory guard and resumed with zero re-runs).

| `T_i`, T1 − A0 paired | pooled (shot CI) | per seed | per discharge |
|---|---|---|---|
| cut | **−0.048** [−0.371, +0.067] | −0.020 / −0.062 / −0.062 / −0.049 (0/4 CI clear) | 31114 +0.02 · 31921 +0.09 · 31902 −0.19 · 31873 −0.61 |
| inclusive | **−0.061** [−0.814, +0.078] | −0.004 / −0.134 / −0.047 / −0.059 (0/4) | 31114 +0.06 · 31921 +0.08 · 31902 −0.86 · 31873 −0.73 |

No gain: two discharges up by a few hundredths, two down by a lot, every CI through zero. The
strata say where: in the **no-raw-cover rows** (307 rows, all #31902 before 1.75 s, where the
feature table is all zeros) T1 − A0 is **−0.23 cut / −2.03 inclusive** — the loader's per-shot
z-scoring turns a block of zeros into a constant far from the covered rows' distribution, so the
`T_i` encoder sees an input it never met in training (no pool discharge has an uncovered labelled
stretch of that length). Inside covered rows the picture is flat: burst-in-window −0.03 / −0.06,
10–50 ms −0.10 / +0.01, > 200 ms +0.08 / +0.14. #31873 (ELM-suppressed, 122 bursts, fully covered)
loses −0.6 / −0.7 on its own — the one discharge where the features are most active is the one
they hurt most. `V_rot` moves by −0.009 / −0.039 although the branch is blind to the extras by
construction (test green): the refit epoch count differs (15 vs 14, 11 vs 19) and the larger
`T_i` LSTM shifts the init RNG stream, so the `V_rot` branch is a different draw, not a leak.

**Reading.** The in-discharge ceiling was real and the cross-discharge transfer failed, for the
same reason §8bd gave for `V_rot`: 27 per-shot-z-scored channels on six training discharges learn
the pool's burst statistics, not a law. The raw stream does carry `T_i`-increment information
(ceiling 10/12) that the 100 Hz snapshot destroys; the delivered twelve cannot teach a model to
use it. Pre-registered expectation (+0.00 … +0.05, gain in the burst-in-window stratum) **not
met**.

**What it does not show, and the named next measurements** (§8j rule). (1) The no-cover artifact
is an implementation fact, not physics: a coverage flag or leaving uncovered rows at the covered
mean would remove the −0.23 / −2.03 stratum — a one-line fix that must be a new registration, not
a re-run. (2) The variance problem is the channel count: a 3-channel physical version (burst in
window, its energy, time since burst) is the natural next registration if more raw discharges
are ever delivered. (3) The decisive lever is the number of discharges, not the feature: the
641-shot grid cannot supply burst energy (snapshots), so a raw-stream `T_i` arm needs a raw
delivery of tens of discharges — the §8j "next measurement" for the `T_i` path, alongside B.16's
event gate, which is the 641-shot proxy for exactly this quantity.

### Addendum to §8bd (2026-10-29) — the NBI torque is a step input, and a step input cannot move a 10 ms increment

§8bd closed with "NBI torque is the one named lever for `V_rot`". 승상님 asked what an NBI channel
would actually add once one notices that KSTAR NBI power is **piecewise constant**: an ion source is
switched on at a programmed time and held at fixed voltage and current through the flat-top, and
the power changes only at a handful of source on/off steps per discharge (the diagnostic beam
notches are 10–50 ms dips). The literature says the same (Ahn 2012 NF, duty-cycle step 1.5 → 1.1 MW;
the EPS 2018 beam-blip study; arXiv 2207.06610 "total NBI power constant except short blips").
Three measurements settle what that implies for this dataset.

1. **Our discharges run the CES beam continuously.** On 55 random shots the `T_i` label fills 94 %
   of a 10 ms grid over a median 4.0 s span, the longest unbroken 10 ms run is 1.3 s (max 6.1 s)
   and gaps > 100 ms are essentially absent — i.e. 2022 H-mode heating operation with NB1A on
   throughout the labelled window, not blip mode (where labels would come in 20 ms bursts every
   0.5 s).
2. **The slow band carries 0.3 % of the 10 ms increment variance.** On 120 random shots (held rows
   removed, contiguous 10 ms runs ≥ 0.6 s), the share of the increment variance `Var(Δy)` carried
   by a 300 ms moving average of the series is `V_rot` **0.34 %** (IQR 0.21–0.94 %, max 2.7 %,
   20 shots with enough genuine rows) and `T_i` 0.25 % (63 shots). Everything a step-held input
   can act on lives in that band; a perfect NBI channel therefore bounds the removable 10 ms
   increment variance at ~1 %. The remaining 99.7 % is intrinsic torque, ELM momentum loss and
   measurement noise, which no 0D channel carries.
3. **One published trace over our own labels** (`experiments/b6_mus/nbi_step_overlay.py`,
   `docs/paper/figures/fig_nbi_step_31359.png`; `P_tot` digitised from Kim et al. 2024 Nat. Commun.
   Fig. 6a, CC BY, PMC10858871, 12.9 ms / 0.08 MW per pixel, NBI + ECH total). #31359 has one
   heating step, 3.5 → 4.0 MW at 4.52 s, and 50 ms notches every 0.5 s. The edge `V_rot` level
   rises from ~107 to a median 143 km/s within 0.2–0.7 s of the step (the momentum confinement
   time scale; the "before" side has one genuine row), then falls from ~150 to ~60 km/s after
   5.5 s **with the heating constant** — the paper's ETB formation and density rise, i.e. even the
   macro level has non-NBI drivers. The 10 ms increments (sd 13 km/s) have the same width before
   and after the step and across every notch.

**Corrected statement.** `V_rot` has two scales. The **macro level** (hundreds of ms, between
discharges) is set by the torque balance — co-NBI torque plus intrinsic torques over density and
momentum loss (Shi et al. 2016 NF 56 016014; Yang et al. 2018 NF 58 066008) — and in a nowcaster
that level is already in the input: the carried observation 10 ms earlier *is* the level. The
**10 ms increment** is set by unobserved intrinsic torque and ELM losses (99.7 % of its variance is
outside the slow band). So the NBI channel would not change the 10 ms `V_rot` result, and the P02
conclusion is final rather than provisional: at 10 ms `V_rot` converges to the history-only OU
filter *even with the driver observed*. The channel matters only if the thesis scope changes to
≥ 100 ms forecasting after power steps or to the cold start before the first CES row; the data
request is deferred on that basis (승상님, 2026-10-29). A transcribed-trace pilot was considered
and not run: with a 0.3 % bound and one discharge, its paired CI (±5–10 %) cannot show anything
either way.

## 8bf. B.17 (2026-10-29): the predictive-distribution head — a two-component mixture is calibrated, beats the oracle-scale Gaussian by +0.2 … +0.44 CRPS and lifts the posterior mean, but by heavy tails rather than two peaks; the Gaussian head is rejected on `V_rot`

**Question and design** (`experiments/PREREGISTRATION_B17.md`, committed 73aa71b before any number;
`experiments/seq/model_seq_v2d.py`, registry `v2g` / `v2m2`; runner `experiments/b17_dist/run_b17.py`;
distribution dump `eval_dist_b17.py`; scoring `score_b17.py`). seq_v2 unchanged up to the heads; the
head emits a conditional distribution — `v2g` a heteroscedastic Gaussian (μ, σ), `v2m2` a
two-component Gaussian mixture (π, μ₁, μ₂, σ₁, σ₂) — trained by the NLL with val-NLL checkpoint
selection. `forward()` returns the posterior mean, so the point harness (eval_seq, paired vs the
frozen backbone, PR4 ladder) scores the arm with zero changes; the distribution is dumped on the
same frozen TEST rows (two pairing guards: the arm's mean and the backbone's point prediction must
both reproduce their npz `se_model`; all 16 runs passed, max deviation 0.87 eV² on a mean SE of
~1.5 × 10⁴). References need no training: **R0** = N(backbone point, σ₀) with σ₀ the backbone's own
TEST RMSE (oracle scale, conservative for us), **R1** = N(arm mean, σ₁) with σ₁ the arm's TEST RMSE
(the heteroscedastic part alone). CRPS closed form (Grimit 2006), shot-cluster paired bootstrap
B = 10⁴ seed 12345, PIT coverage, Spearman(σ, event energy / staleness), and the bimodality test
(min π ≥ 0.1 ∧ |μ₁ − μ₂| ≥ 2 max σ). Unit tests `tests/test_seq_v2d.py` (6: mean path = decoded
mean, NLL = closed forms, `V_rot` parameters bit-identical under fast-channel perturbation,
causality, init σ = 1, CRPS closed form = Monte Carlo).

**Batch log.** 16 runs (2 arms × 2 populations × splits 42/1/7/123), epoch cap 60; **7/16 runs
hit the cap** (`v2m2` 6/8: cut s42/s1/s123, incl s42/s1/s123; `v2g` incl s1) — the mixture NLL
is still improving on val at 60 epochs, so its numbers below are a lower bound on what the arm
does (recorded as §2 of the registration requires; the cap is a registered constant and was not
raised). ~10 min per run (train 2.5–7 min + TEST eval 3.5–4.6 min + dump 0.4 min). The batch
was interrupted twice: once by 승상님 at 3/16 and once at 12/16 by the host memory guard (another
project's 8 GB process); resumed from the artifacts both times. One implementation fact learned
the hard way: the frozen B.1/B.5 control npz predate the `_row` key, so the dump pins control rows
by (shot, dt, y_true). The Claude Code tool sandbox hides the GPU (`torch.cuda.is_available()`
False while the device is idle); GPU jobs run with the sandbox off.

**Results — `T_i`** (CRPS skill = 1 − ΣCRPS_arm/ΣCRPS_ref; `*` = CI excludes 0; cov = central
50 % / 90 % PIT coverage; MSE guard = posterior-mean paired MSE skill vs the backbone, `arm sig /
control sig` out of 4):

| pop | arm | CRPS vs R0 per split | pooled vs R0 | pooled vs R1 | cov 50 % | cov 90 % | MSE guard (4 splits) |
|---|---|---|---|---|---|---|---|
| cut | `v2g` | +0.147\* +0.154\* +0.042\* +0.144\* | **+0.122** [+0.097, +0.149]\* | +0.138\* | 0.62–0.66 | 0.93–0.94 | +0.019 +0.029 **−0.088\*** +0.001 (2/1) |
| cut | `v2m2` | +0.212\* +0.202\* +0.216\* +0.228\* | **+0.215** [+0.189, +0.242]\* | +0.197\* | **0.51–0.56** | **0.90–0.92** | **+0.022 +0.015 +0.051\* +0.026** (1/0) |
| incl | `v2g` | +0.247\* +0.192\* +0.215\* +0.218\* | +0.216 [+0.174, +0.259]\* | +0.234\* | 0.74–0.80 | 0.95–0.97 | +0.006 −0.011 −0.032\* −0.054\* (0/2) |
| incl | `v2m2` | +0.431\* +0.408\* +0.457\* +0.452\* | **+0.438** [+0.404, +0.474]\* | +0.428\* | 0.53–0.56 | 0.89–0.92 | +0.014 +0.012 +0.017\* −0.003 (1/0) |

`v2m2 − v2g` paired CRPS, pooled: cut **+0.106** [+0.093, +0.119]\*, incl **+0.283** [+0.261, +0.305]\*.
Absolute CRPS (cut s42): `v2m2` 54.8 eV, `v2g` 59.4, R0 69.6, R1 69.0; NLL 5.68 / 6.05 / 6.48 / 6.47.
By stratum (cut s42): `v2m2` CRPS vs R0 +0.18 on peak rows and +0.22 on non-peak rows; its
posterior-mean MSE vs backbone +0.01 / +0.02.

**Results — `V_rot`:**

| pop | arm | CRPS vs R0 per split | pooled vs R0 | pooled vs R1 | cov 50 % | cov 90 % | MSE guard (4 splits) |
|---|---|---|---|---|---|---|---|
| cut | `v2g` | **−0.293** +0.078 +0.180\* +0.203\* | +0.084 [+0.015, +0.149]\* | +0.120\* | 0.79–0.90 | 0.97–0.99 | **−0.093\* −0.030\* −0.054\*** −0.057 (0/**3**) |
| cut | `v2m2` | +0.191\* +0.268\* +0.438\* +0.391\* | **+0.342** [+0.290, +0.394]\* | +0.339\* | 0.57–0.63 | 0.91–0.96 | −0.004 +0.029 −0.011 −0.009 (0/0) |
| incl | `v2g` | −0.444 +0.019 +0.115\* +0.149\* | +0.009 [−0.091, +0.083] | +0.053 | 0.83–0.89 | 0.96–0.98 | −0.083\* −0.031 −0.071\* −0.030 (0/2) |
| incl | `v2m2` | +0.218\* +0.241\* +0.437\* +0.410\* | **+0.345** [+0.295, +0.396]\* | +0.342\* | 0.60–0.62 | 0.91–0.96 | +0.020 +0.009 −0.015 −0.005 (0/0) |

`v2m2 − v2g` paired CRPS, pooled: cut +0.282\*, incl +0.340\*.

**Physics readout (H17-b, `v2g`).** `T_i`: Spearman(σ, fast-channel increment energy)
+0.16 / +0.17 / +0.20 / +0.16 (cut), +0.10 … +0.17 (incl); σ on event rows / non-event rows
1.16–1.22 (cut), 1.20–1.31 (incl). The sign and the ordering are the registered ones but the bars
(+0.2 and 1.3 on 4/4) are missed. `V_rot`: Spearman(σ, staleness) **+0.00 … +0.01 on 8/8** — the
learned `V_rot` σ does not grow with the time since the last observation at all; Spearman(σ, energy)
−0.15 … +0.05 (the `V_rot` branch cannot see the energy; the residual correlation runs through
shot level). Median predictive sd (physical): `T_i` 83–107 eV cut (`v2g`), 97–120 eV (`v2m2`) against
an oracle homoscedastic 156–202 eV; `V_rot` 14–23 km/s (`v2g`) against 25–41 km/s.

**What the mixture actually learned (the finding the registration did not anticipate).** The
bimodality test fires on **0 rows** (cut `T_i`: 1 row in 32,589 on s42; `V_rot` 29–62 rows,
≤ 0.6 %). The second component is a *wide* one with small weight — min π median 0.07 (`T_i`),
0.10 (`V_rot`); σ₂ 90th percentile 545 eV / 221 km/s against a first-component median of 34–138 eV;
|μ₁ − μ₂| / max σ median 0.48 — i.e. `v2m2` is a **scale mixture**, a heavy-tailed one-peak
distribution, not two peaks. That is why it is calibrated where the Gaussian is not: `v2g` covers
62–66 % of rows with its central 50 % interval (too wide in the bulk) while hitting 93 % at 90 %,
exactly the signature of a Gaussian whose σ is inflated by tails it cannot shape; `v2m2` sits at
51–56 % / 90–92 %. It is also why `v2g` loses on `V_rot` MSE (3/4 significant, −0.03 … −0.09): the
Gaussian NLL's 1/σ² weighting collapses on the |v| > 1000 km/s rows (§8ab) and drags the mean,
whereas the mixture parks those rows in the wide component at ~10 % weight and leaves the mean
alone (0/4 significant on either side, a tie). The same mechanism lifts the `T_i` posterior mean
above the MSE-trained backbone on 4/4 cut splits (+0.015 … +0.051) — the mixture NLL is, in effect,
the robust loss B.14 looked for (§8ay: Huber moved the median; here the heavy-tail likelihood moves
the mean and the calibration together).

**Hypotheses (§4, verbatim).** H17-a (`v2g` `T_i` CRPS vs R0 on ≥ 3/4 splits, pooled vs R0 and vs R1):
**PASS** in both populations (4/4, pooled +0.122\* / +0.216\*, vs R1 +0.138\* / +0.234\*). H17-b
(σ physics): **FAIL** on the bars for both targets (`T_i` ρ 0.16–0.20 < 0.2 on 3/4, ratio < 1.3 on
3/4; `V_rot` ρ(σ, staleness) ≈ 0). H17-c (`v2g` calibration, 90 % ∈ [0.85, 0.95] ∧ 50 % ∈ [0.45, 0.55]
on ≥ 3/4): **FAIL** 0/4 in both populations — the 50 % band is 0.62–0.80. H17-d (`v2m2 − v2g` CRPS
pooled CI > 0 ∧ bimodal-row enrichment ≥ 2 on ≥ 3/4): **FAIL** — the CRPS half passes (+0.106\*,
+0.283\*), the enrichment is undefined (no bimodal population). H17-e (MSE guard): `v2m2` **PASS**
on both targets and both populations (0/4 significant losses, pooled means +0.028 / +0.010 `T_i`,
+0.001 / +0.002 `V_rot`); `v2g` **FAIL** — `V_rot` 3/4 (cut) and 2/4 (incl) significant losses,
and incl `T_i` 2/4 (cut `T_i` alone would pass: 1/4, mean −0.010).

**Rules (§4, verbatim).** 1. "Calibrated nowcaster" claim (H17-a ∧ H17-c ∧ H17-e on `v2g`): **not
supported** in either population, because H17-c fails and `v2g` is rejected. 2. Reported arm: H17-d
fails → `v2g` by the letter, which rule 4 then rejects. 3. σ narrative: "σ is unexplained
heteroscedasticity" by the letter; measured, it rises 1.2× on event rows and with the event energy
(ρ ≈ +0.17) and does not rise with staleness. 4. `v2g` is **rejected** in both populations on the
`V_rot` guard, cause recorded above (NLL weighting on the `V_rot` tail rows; cut s42 non-peak
stratum paired MSE −0.098, peak −0.064).

**Verdict and what the record says beyond the rules.** The registration tied the claim to the
Gaussian arm and the mixture to a bimodality test, and the data answered a different question:
the error of the nowcast is **one-peaked and heavy-tailed**, not two-valued, and a two-component
scale mixture is the arm that is calibrated on `T_i` (50 % / 90 % coverage 0.51–0.56 / 0.89–0.92 on
8/8; on `V_rot` 90 % is in range 8/8 and the 50 % band is slightly wide, 0.57–0.63), beats the
oracle-scale homoscedastic reference by **+0.215 / +0.438** (`T_i`) and **+0.34 / +0.35** (`V_rot`)
in CRPS with 16/16 splits individually significant, and improves the posterior mean on `T_i`
(cut 4/4 positive, pooled +0.028; H16-c-style guard passed) while tying on `V_rot`. By the
pre-registered letter this is **exploratory**: the rules name no arm and the claim is not made.
The physics reading of the fail on H17-b is informative in itself: the `T_i` predictive sd carries
the event signal at the size the §8bb p_obs proxy predicted (1.2–1.3×, enrichment 1.27–1.34× in
§8bb) but not more, and the `V_rot` sd ignores staleness — the learned uncertainty is a per-shot
scale, not the OU variance growth the Kalman cell encodes (B.16's σ_n / σ_s), which is a second
sense in which the `V_rot` path is history-only.

**Named next measurement** (§8j rule). The mechanism is heavy tails; the parametric form that
states that hypothesis directly is a **Student-t head with learnable ν** per target (one variable:
the likelihood family; mixture → t). B.17b registers `v2t` against `v2m2` as the control, with the
H17-c/H17-e bars as written here, a ν readout (ν ≈ 2–4 expected from B.14's ν = 3 and the §8aq
tail share), and the "calibrated nowcaster" claim bound to whichever arm the registration names
*before* numbers. Its passing would make today's result confirmatory; its failing would say the
mixture's gain is specific to a free second scale, which the thesis can still report as the
calibrated arm of record under the exploratory label. Artifacts: `data/.b17_{v2g,v2m2}_{cut,incl}_s{42,1,7,123}/`
(point artifacts + `dist_test.npz`), `data/.b17_runner_summary.json`, `data/.b17_summary.json`.

## 8bg. B.16 executed (2026-10-27 … 29): the Kalman anchor moves the `T_i` median below persistence for the first time (1.02× → 0.98× cut, 1.34× → 1.09× inclusive) and lifts the quiet-discharge win rate to the predicted 0.50, but it reaches +0.04 of the +0.05 bar in the cut population, the event gate is negative, and `V_rot` is untouched

**Question and design** — §8bc and `experiments/PREREGISTRATION_B16.md` (committed 2026-10-13 before
any number). Output `ŷ = a + g·r`: a learnable scalar OU–Kalman cell per target as the anchor `a`
(initialised at the §8 OU fits, `(ρ, σ_s, σ_n, α)` learnable), the seq_v2 LSTM as the residual `r`
(head zero-initialised, so training starts exactly at the Kalman filter), and an event gate `g` on
the fast-channel increment energy (`v4k`) or `g ≡ 1` (`v4kng`). Control = the frozen B.1 / B.5
backbone runs. Readouts: paired MSE vs the backbone (runner), M1–M5 via `rescore_b13.py --extra`
(`data/.b16_rescore.json`), the §8an quiet-tercile discharge win rate vs the causal GP pooled over
the four splits, `v4k − v4kng` paired M1, the learned physics parameters. Verdict script
`experiments/b16_physics/verdict_b16.py` → `data/.b16_verdict.json`.

**Batch log.** 16 runs, epoch cap 60, none bound (max 30; best epochs 4–24). **~20–25 min per
run** (12–19 min training + 3–5 min TEST eval): the Kalman cell is a per-step Python loop, and the
registration's 1.5–3 min estimate was wrong by 10×. The batch was interrupted four times —
once by 승상님 for a mid-check at 3/16, three times by the host memory guard (other projects'
processes, not this batch) — and resumed from the artifacts each time; the CPU rescore (~2.5 h)
was killed three times before it completed. Total wall ≈ 5 h GPU + 2.5 h CPU.

**Results — `T_i`** (M1 = median-AE skill vs the backbone, `*` CI excludes 0; medAE/pers = the
arm's median |e| over persistence's, pooled 4 splits; quiet = win rate vs the causal GP in the
quiet tercile of discharges, pooled; MSE paired = runner, `+`/`−` significant):

| pop | arm | M1 vs backbone per split | M1 pooled | medAE / pers (backbone) | quiet / mid / variable win (backbone) | MSE paired vs backbone |
|---|---|---|---|---|---|---|
| cut | `v4k` | +0.041\* +0.043 +0.022 +0.042\* | **+0.038** [+0.019, +0.057] | **0.985** (1.024) | **0.50** / 0.84 / 0.87 (0.43 / 0.85 / 0.86) | −0.000 −0.008 +0.033⁺ +0.030, mean +0.013 |
| cut | `v4kng` | +0.015 +0.073\* +0.058\* +0.033\* | **+0.045** [+0.030, +0.062] | **0.978** | **0.54** / 0.77 / 0.86 | −0.010 +0.013 +0.038⁺ +0.042⁺, mean +0.021 |
| incl | `v4k` | +0.079\* +0.212\* +0.087\* +0.185\* | **+0.149** [+0.097, +0.197] | **1.138** (1.337) | **0.26** / 0.55 / 0.81 (0.13 / 0.56 / 0.79) | +0.013 +0.023 +0.010 −0.016⁻, mean +0.008 |
| incl | `v4kng` | +0.166\* +0.262\* +0.066\* +0.222\* | **+0.186** [+0.138, +0.233] | **1.089** | **0.35** / 0.56 / 0.79 | −0.002 +0.004 +0.004 −0.003, mean +0.001 |

Aggregate skill vs PCHIP (the registered "tie ±0.02"): cut backbone +0.243 pooled, `v4k` +0.254,
`v4kng` +0.261; inclusive +0.278 / +0.281 / +0.278 — a tie as predicted. Overall discharge win rate
vs the causal GP: cut 0.714 → 0.736 / 0.725; inclusive 0.495 → 0.543 / 0.568. Quantile ladder
`v4kng` vs backbone (cut): q50 0.955, q90 1.002, q99 0.968 — the gain is the bulk, the tail is
untouched (tail share 0.54 / 0.54). By stratum (M1 vs backbone, 4 splits): event rows +0.04 …
+0.12, non-event rows +0.01 … +0.07, **rows with stale history (Δt > 15 ms) −0.11 … −0.28 on 8/8
cells** — the anchor's propagation toward the slow level `m` costs the median exactly where the
last observation is old (the MSE on those rows is +0.03, the tail is helped, the bulk is not).
`v4k − v4kng` paired M1: cut −0.008 [−0.019, +0.003], inclusive **−0.045** [−0.059, −0.033]; quiet
win rate `v4k − v4kng` −0.04 / −0.09.

**Results — `V_rot`:** M1 vs backbone `v4k` cut −0.043 +0.124\* −0.040 −0.038 (pooled +0.005
[−0.029, +0.038]), inclusive −0.066 −0.017 +0.066\* +0.079\* (pooled +0.028 [−0.011, +0.064]);
`v4kng` pooled −0.008 / +0.034, CIs through 0. medAE / persistence **1.162 / 1.177** (backbone
1.168) in cut, 1.159 / 1.151 (1.191) in inclusive — the predicted 0.92–1.01 was **not reached**.
MSE paired vs backbone negative in all four arm × population cells (−0.044 … −0.063 pooled; s42
−0.14 … −0.16, 1/4 significant loss each; the s42 loss is the peak stratum, paired MSE −0.83 on
event rows vs −0.03 on the rest). Quiet win rate 0.36 → 0.41 / 0.26 (cut), 0.29 → 0.30 / 0.33
(inclusive); overall 0.466 → 0.452 / 0.411.

**Physics parameters (H16-f).** Learned τ: `T_i` 83–94 ms (cut) / 80–85 ms (incl), `V_rot` 85–93 ms;
σ_n and σ_s within 0.92–1.09× their OU-fit initial values on 16/16 — the cell stayed a physical OU
filter (σ_n ≈ 110 eV cut / 190 eV incl, 30 km/s), and it also barely moved: with the residual
absorbing the gradient, the cell is the §8 fit it was initialised at.

**Hypotheses (§4, verbatim).** H16-a (`v4k` `T_i` M1 ≥ +0.05 ∧ CI > 0 on ≥ 3/4): **cut FAIL** (0/4;
+0.041 +0.043 +0.022 +0.042), **inclusive PASS** (4/4). H16-b (quiet win rate ≥ backbone + 0.05):
**PASS** both (cut 0.43 → 0.50, Δ +0.07; incl 0.13 → 0.26, Δ +0.13). H16-c (guard): **PASS** for
both arms in both populations (significant losses 0/4, 0/4, 1/4, 0/4; pooled means +0.013,
+0.021, +0.008, +0.001). H16-d (`V_rot`): **FAIL** both (1/4 and 2/4 splits at the M1 bar, 1/4
significant MSE loss each). H16-e (gate buys ≥ +0.02 M1 or +0.05 quiet): **FAIL** both — the gate
is **negative** (−0.008 n.s. cut, −0.045\* inclusive; quiet −0.04 / −0.09). H16-f: **PASS**.

**Rules (§4, verbatim).** 1. Physics rung adopted = H16-a ∧ H16-c in the **cut** population: **not
adopted** (H16-a fails by 0.01 in cut). The rule defines adoption on cut and names "unconditional"
only when the inclusive population also passes; it has no label for "inclusive passes, cut does
not", which is what happened — recorded here as an **inclusive-conditional rung** by analogy with
B.5's "cut-conditional", with the asymmetry noted as a registration gap. 2. Backbone replacement:
**no** (H16-d fails; `T_i` MSE sign 4/4 positive in neither population). 3. H16-e fails → **`v4kng`
is the reported arm and the gate is recorded negative.** 4. No arm is rejected (no cell with ≥ 2/4
significant losses).

**Verdict.** The derivation's structural prediction was half right and precisely measurable.
*Right*: putting the OU–Kalman anchor under the output moves the `T_i` median across the
persistence line for the first time on record (cut 1.024× → 0.978×; inclusive 1.337× → 1.089×, where
the K < 1 shrinkage of spike anchors predicted in §3 is exactly what happens), lifts the
quiet-discharge win rate to the predicted ≥ 0.50 (0.43 → 0.50 / 0.54), does so at zero aggregate
cost (`T_i` MSE +0.01 … +0.02, PCHIP skill tie), and keeps physical parameters. *Short*: the cut
median lands at 0.98×, not the 0.875–0.94× the filter bound and the Huber loss (§8ay, 0.94×)
promised, so the structural route recovers **~30 % of the predicted headroom** and less than the
loss route did (M1 +0.04 vs Huber's +0.08 … +0.10). *Wrong*: the event gate — the (M6) prediction
that opening the residual only on events removes the quiet-row variance cost — is negative in both
populations; the ungated residual is better everywhere, including on quiet discharges (0.54 vs
0.50). The likely mechanism, not proven here: the gate starts at σ(−2) = 0.12 and throttles the
residual's gradient while early stopping (patience 6, best epochs 5–24) ends training before it
opens, so `v4k` is `v4kng` under-trained; a gate initialised open (bias +2) is the one-line test.
`V_rot` is untouched in the median (1.16× vs 1.17×) and slightly worse in aggregate: the residual
head re-introduces in the bulk what the anchor removes, and the `V_rot` gate, seeing only the slow
state, has nothing to close on — consistent with §8bf's finding that the learned `V_rot`
uncertainty carries no staleness dependence either.

**What the record says beyond the rules, and the named next measurements** (§8j rule). (1) The
stale-row stratum (Δt > 15 ms, M1 −0.11 … −0.28 on 8/8) is where the remaining bulk headroom sits:
the anchor relaxes toward the slow level `m` at rate ρ while the truth does not, which is the
OU model's own failure on long gaps (§8ar: the increment distribution is bursty, not diffusive).
A level-holding anchor (ρ → 1 on the propagate step when unobserved, i.e. persistence of the
filtered value) is the one-parameter variant to register. (2) The pure learned anchor with the
residual removed is the P02 estimator itself; scoring it from the `v4kng` checkpoints with `r ≡ 0`
(inference only, 8 evals) would say whether the 0.921× `V_rot` bound is reachable by the filter
alone or whether the bound is wrong — a new registration, because it is a new TEST readout. (3)
The gate re-run with an open initialisation is a B.16 sensitivity row, 8 runs, not a new
hypothesis. (4) For the thesis the sentence is: *the structure the derivation asked for exists,
is physical, and buys a third of what the derivation promised; the rest of the bulk headroom is
the long-gap regime, where the OU prior is the wrong prior.* Artifacts: `data/.b16_{v4k,v4kng}_{cut,incl}_s{seed}/`,
`.b16_summary.json`, `.b16_rescore.json`, `.b16_verdict.json`.

## 8bh. B.14 sensitivity row (2026-10-29): with the checkpoint chosen on the arm's own objective, the inclusive robust arms train 30–60 epochs instead of 8–23 and the median gain doubles — but one split keeps a significant `T_i` MSE deficit, so by the registered letter the rung stays cut-conditional

**Design** (`experiments/PREREGISTRATION_B14.md` §9, registered 2026-10-12 before any number).
One variable against §8ay's inclusive runs: early stopping, checkpoint and LR plateau read the
arm's own val objective (`CES_SEQ_SELECT=loss`) instead of val MSE. Inclusive population only,
huber / tnll × 4 splits = 8 runs (`data/.b14s_{arm}_incl_s{s}`, summary `.b14s_summary.json`),
co-metrics `rescore_b13.py --pops incl --extra huber_s=… tnll_s=… --out data/.b14s_rescore.json`.
(The runner was briefly started without `--pops incl` and had begun one cut run; it was stopped
and that partial directory deleted, so the row is exactly the registered 8 runs. The rescore was
killed twice by the host memory guard and completed on the third start.)

**Training.** huber epochs run / best 55/49, 31/25, 30/24, 60/59; tnll 36/30, 60/60, 60/60, 60/59
(§8ay with MSE selection: 8–23 / 2–17). Four of eight runs reach or nearly reach the 60-epoch
cap — the robust objective is still improving there.

| arm / target | M1 vs MSE backbone (per split) | M1 pooled | MSE paired vs backbone (sig. worse / better) | median AE / persistence (backbone) |
|---|---|---|---|---|
| huber `T_i` | +0.224\* +0.361\* +0.214\* +0.319\* | **+0.285** [+0.250, +0.322] | −0.009 −0.003 −0.014 **−0.040\*** (1 / 0), pooled −0.019 [−0.037, 0.000] | **0.956** (1.337) |
| huber `V_rot` | +0.101\* +0.051\* +0.140\* +0.246\* | **+0.145** [+0.108, +0.181] | +0.030\* +0.028 −0.022 −0.014 (0 / 1), pooled −0.005 | **1.019** (1.191) |
| tnll `T_i` | +0.261\* +0.376\* +0.244\* +0.328\* | +0.308 [+0.272, +0.343] | −0.044 −0.026 −0.027 −0.076\* (1 / 0), pooled −0.046 | 0.926 |
| tnll `V_rot` | +0.100\* +0.058\* +0.157\* +0.247\* | +0.151 [+0.119, +0.189] | +0.013 +0.026 −0.025 −0.022 (0 / 0), pooled −0.011 | 1.011 |

Against §8ay's MSE-selected inclusive runs the `T_i` M1 rises from +0.157 to **+0.285** (huber) and
from +0.084 to +0.308 (tnll), the `V_rot` M1 turns from −0.034 / −0.150 to **+0.145 / +0.151**
(4/4 significant), and the `T_i` MSE deficit shrinks from −0.030 pooled (2/4 significant) to −0.019
(1/4, CI touching 0). Medians: huber puts the inclusive `T_i` median at 0.956× persistence (the
backbone sits at 1.337×; B.16's anchor reached 1.089× in this population, §8bg), ties the causal GP
in the median (+0.023 [−0.012, +0.057]; the backbone is −0.367 there), and keeps the PCHIP MSE skill
at +0.218 … +0.289 (backbone +0.225 … +0.316). `V_rot` median 1.02× persistence — inside the
0.92–1.01 band P02 calls the optimal causal filter, the first arm on record to reach it, and a tie
with the causal GP (+0.010).

**Reading (§9, verbatim).** Case 1 requires huber MSE significant deficits 0/4 **and** M1 ≥ 3/4.
M1 is 4/4 on both targets; the MSE guard has one significant `T_i` deficit (s123, −0.040). By the
letter this is **case 2: the rung stays conditional on the cut population** (§8ay unchanged), with
the note that the case-2 sentence ("the spike rows are information the MSE model actually fits")
now rests on a pooled deficit of −0.019 — inside the project's ±0.02 tie width — carried by one
split. Case 3 is excluded outright: the inclusive median gain was not an under-training artifact;
it grew when training lengthened.

**What the row says beyond the rule.** The inclusive failure of §8ay was mostly the selection
rule, as §9 suspected: under a matched rule the robust arm improves the typical row of both
targets in the population with the fit-failure spikes left in, and gives up at most 0.02–0.04 of
aggregate skill on one split for it. For the thesis, the honest sentence is: *the noise-floor
Huber loss with its own checkpoint rule is the arm that reaches the P02 median band for `V_rot`
and the persistence line for inclusive `T_i`; whether that is a rung in both populations is
decided by one split's MSE guard and is recorded as cut-conditional.* Named next measurement: the
same row with the epoch cap at 100 (four of eight runs are cap-bound), and the cut-population
counterpart with `--select loss`, so that both populations use the same rule — one 16-run batch,
to be registered before it is run.
