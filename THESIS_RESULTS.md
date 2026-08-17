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

## 9. Recommended framings for the thesis

1. **Lead with `CES_TI` beating offline interpolation** — it passes PR4 on four independent test
   splits, including three never touched by model selection. This is the headline.
2. **Lead the practical case with causal superiority.** The nowcaster substantially outperforms every
   causal CES gap-filler; for real-time use the comparison to future-using interpolation is generous
   to the baseline and the model still wins on `CES_TI`.
3. **Report `CES_VT` as an honest tie and foreground the asymmetry as the scientific contribution.**
   The physics story (fast diagnostics inform core temperature but not toroidal rotation at 10 ms) is
   a genuine, interpretable finding, and the ablation makes the mechanism explicit.
4. **The top lever for `CES_VT` is data, not architecture: get NBI torque/power into the dataset.**
   It is the actual driver of toroidal rotation and it is currently absent.
5. **Planned refinement — bracket-distance stratification.** Stratify by distance to the nearest
   *future* anchor rather than by Δt-since-last-observed alone; this should isolate the regime where
   the nowcaster's fast-diagnostic information has the most marginal value over interpolation.

---

*All quantitative claims above were regenerated on 2026-07-14 from the checkpoints listed in the
provenance table, using the repository's own harness (`compare_baselines.py`, `bootstrap_compare.py`,
`peak_analysis.py`, `evaluate.py`) with the architecture pinned as described. The model neither
denormalizes internally nor was tuned for this report.*
