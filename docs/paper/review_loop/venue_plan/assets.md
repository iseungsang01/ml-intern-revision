# Venue plan: asset inventory (2026-11-15)

Scope: what the repository already holds that could be developed into a main-venue submission
(NeurIPS main track or *Nuclear Fusion*), what is feasible on the local RTX 5060 Laptop (8 GB),
and what needs new data. Read-only pass: no model was trained, no other file was edited.

Sources read: `CLAUDE.md`, `THESIS_RESULTS.md` §0 and the §8 index, `ces_prediction/experiments/README.md`,
`PROJECT_KNOWLEDGE.md` (headings + geometry block), `docs/paper/review_loop/round_1.md`, `round_2.md`,
`docs/paper/short/main_short.tex`, `docs/theory_*.md` (outlines), `docs/b28_weight_physics_report.md`.
Data statements marked **[probed today]** come from two throwaway scripts run on the real files
(not committed; the heuristics are stated and are unvalidated against any reference signal).

---

## 1. Data on disk

### 1.1 The 641 shot CSVs (`data/s{shot}.csv`)

| item | value | source |
|---|---|---|
| files / shots | 641, shot numbers 30801–32751 (KSTAR 2022 campaign), one file per shot | [probed today] |
| header | identical in all 641 files: `time, CES_TI, CES_VT, BES_0105..0307 (9), ECEI_GT1103/1203/1303/1403 (4), MC1T03, MC1T16` = 18 columns | [probed today] |
| rows | 247,207 on a 10 ms grid; per file 3 / 339 / 1,483 (min / median / max) | [probed today] |
| usable length | after dropping the provider's first/last padding rows (t = 0 and t = 30 s): 595 files have at least 50 rows; median block 4.0 s, max 14.8 s | [probed today] |
| irregular files | 617 files have a 10 ms median step; 24 have coarse or near-empty grids | [probed today] |
| labels | `CES_TI` NaN 8.2 %; `CES_VT` NaN 23.9 % + held (bit-identical repeats) 41.1 % = 65.0 % effectively missing; 3 files have no `T_i` label, 65 files have no `V_rot` label; held runs in 499 / 641 files | `THESIS_RESULTS.md` §1.1, [probed today] |
| fit failures | `CES_TI` > 3 keV: 1,197 rows (0.53 %); file maxima reach 15 keV (median file maximum 2.5 keV) | §8q, [probed today] |
| geometry | one CES chord (channel 18, R = 2.198 m); all 13 BES/ECEI channels within about 1.5 cm of it; Mirnov = two coils of one poloidal array | `PROJECT_KNOWLEDGE.md` l.1187–1196 |

Every shot has the same diagnostics: there is no per-shot diagnostic subset to exploit.

**Not in the data at all:** NBI power / beam on-off waveforms, EFIT (r/a, q95, shape), D-alpha,
line-averaged density, plasma current, stored energy, RMP / ECH waveforms, Thomson, any CES chord
other than channel 18, per-exposure CES fit quality (chi-square, photon counts, fit error bars),
CES frame timing. The only non-CSV shot information is the set of publications in
`data/shot_papers/` (7 PDFs covering 7 of the 12 raw shots) and one digitized NBI power trace
(`ces_prediction/experiments/b6_mus/ptot_fig6a_digitized.csv`, #31359 only).

### 1.2 Raw microsecond streams (`data/raw_mus/`, 10 GB, 36 CSVs)

12 shots × 3 diagnostics (`delivery_manifest.json`):

| shots | BES | ECEI | MC |
|---|---|---|---|
| 31097, 31114, 31357, 31359, 31747, 31873, 31902, 31921, 31923, 32027, 32092, 32097 | 9 ch at 500 kHz (31747 at 1 MHz) | 4 ch at 500 kHz | 2 coils at 2 MHz |

Windows are 3–8 s per shot and cover 99.3–99.9 % of the labelled 10 ms rows, except #31902
(66.7 %) and the #31873 Mirnov file (2 s, 26 %). Roles are frozen in
`experiments/hires_shots/FINAL_12.csv`: test 4 / pool 6 / comparison 2. Positive controls already
passed for Mirnov (#32092 EHO at 4.39 / 8.91 kHz, #31921 envelope jump) and failed for BES at the
registered windows (§8bd).

### 1.3 Which physics events can be tagged from what we have

| event | from the 100 Hz files | from the 12 raw shots | status |
|---|---|---|---|
| Edge bursts (ELM-like) | Weak. The 100 Hz values are raw snapshots, so a burst shorter than 10 ms is caught only by chance. Robust-z > 8 increment spikes: about 1,800 ECEI rows, 1,550 BES rows, 3,800 Mirnov rows out of 245 k; 88 files have > 1 % ECEI spike rows [probed today]. | Yes. `BES_0206` 1 ms variance, robust z ≥ 8: #31873 157 onsets in 7.7 s (20 Hz, median spacing 14 ms, 131 of 766 grid rows contain one); #32092 68 onsets in 6.5 s [probed today]. §8be used the same kind of detector. | Detector exists, **identity unverified**: there is no D-alpha to confirm that a burst is an ELM. Published context exists for #31873 (ELM suppression), #32092 (QH-mode with an ELM-free phase), #31921/31923 (I-mode / FIRE). |
| Beam-off rows | Yes, as a proxy. BES is beam emission; rows with mean BES below 20 % of the file median occur in 184 files (0.63 % of rows; 18 files > 5 %). `T_i` is NaN on 59.7 % of those rows vs 7.4 % elsewhere [probed today]. | same | New observation. §8at looked for a periodic duty cycle and found none, but did not condition missingness on the BES level. Unvalidated without the NBI log. |
| BES level steps (NBI power step, density step, L-H candidate) | Candidates only: a 15 % change of the 100 ms median BES level occurs 2,805 times in 444 files [probed today]. | same | Cannot be attributed (NBI step vs L-H vs density) without NBI / D-alpha / density. |
| L-H / H-L transition | No reliable tag. | Only via times quoted in the papers (e.g. #31357 ETB at 5.5 s). | Needs D-alpha. |
| Sawteeth | No. ECEI channels sit at R ≈ 2.19 m (outer region), not across the inversion radius; 100 Hz snapshots alias a sawtooth period. | Possibly as edge heat pulses in 500 kHz ECEI; never tested. | Low value at this radius. |
| MHD modes (EHO, NTM, kink) | No (aliased; §8b.2). | Yes: 2 MHz Mirnov spectrograms; EHO reproduced on #32092. | Done as a positive control; adding the features lowered `V_rot` skill (§8bd). |
| CES fit failure | Value proxies only: `T_i` > 3 keV; `|V_rot|` > 1,000 km/s (set in review, post hoc). | same | Needs per-exposure fit quality. |
| "The target moves" (discharge-level spread, row-level peak flag) | Yes, from the labels themselves (§8an, §5.1). | same | Already used; label-derived, so not an independent event tag. |

Bottom line for event-conditioned evaluation: on the 641 shots only label-derived strata and
unverified input-activity proxies are available; a physically named event stratum (ELM, L-H,
NBI blip) is possible on at most 12 shots and still lacks a reference signal.

---

## 2. Existing results that are stronger or more general than the 8-page paper uses

Solidity scale: **A** = registered and replicated across populations/partitions; **B** = registered
or replicated but conditional (one population, or a disclosed post hoc element); **C** = exploratory.

| # | result | where | solidity | what the paper uses | what it could become |
|---|---|---|---|---|---|
| 1 | **Noise floor / √2.** The 10 ms `T_i`/`V_rot` scatter ratio is 1.395–1.532 on 8/8 split × population cells, the √2 of a Gaussian line fit; the two channels' noise is independent in size and sign. The √T_i scaling check failed (slope 0.087–0.228 against 0.5 ± 0.15, 0/8). Order-4 difference bound 129.9 eV / 16.4 km/s, bulk MAD 46.4 eV / 4.3 km/s; the top 1 % of 4th differences carries 46.6 % / 65.6 % of the squared mass; the variogram is nugget-dominated (γ(1) > γ(2)). | §8aq, §8bs (B.26), `docs/theory_foundations.md` §1 | B (ratio passes, scaling fails; "photon" wording withdrawn in round 1) | one sentence | A label-noise ceiling analysis: how much of a nowcast error is the target's own irreproducibility, with a model-free estimator. Transfers to any sparse spectroscopic target. For NF it needs per-exposure fit errors to turn the ratio argument into a measurement. |
| 2 | **Tail-vs-median dual readout.** Median-AE skill vs PCHIP is negative on 16/16 cells (cut `T_i` −0.198 [−0.259, −0.135]); the MSE margin starts at the 70th error percentile. A training-free hold-or-jump filter with Student-t noise reaches 0.906–0.920× persistence at the `T_i` median on 8/8 cells (§8bp); Huber at the noise scale moves the median 8–13 % with the MSE interval including 0 in the cut population (§8ay, §8bu). | §8ax, §8ay, §8bh, §8bm–§8bp, §8bu | A for the diagnosis; B for the remedies (population-conditional) | Limitations paragraph + one abstract sentence | A methodology contribution: MSE skill in this regime is a statement about fewer than 1 % of rows; mean and median estimators are different objects (LSTM for the mean, physics filter for the median). Evidence spans ten batches (B.13–B.26). |
| 3 | **Fit-failure anchors decide the interval (A1).** 6–9 rows of about 71,700 per repeat carry 30–57 % of the PCHIP SSE; removing them moves `V_rot` vs PCHIP from +0.313 [+0.072, +0.701] to an interval that includes 0. Symmetric 1,000 km/s rule: vs persistence +0.265 [+0.198, +0.326], vs causal GP pooled +0.151 [+0.038, +0.290] (per split 1/4 cut, 2/4 inclusive). | `review_loop/analyses/A1`, `docs/paper/vrot_failfit_sensitivity.json`, round_1 F5, round_2 G1–G2 | B (rule is post hoc, applied at scoring only) | Claim 2 | A general warning for skill scores against interpolants on fit-derived targets; pairs with #2. Becomes solid once the rule is registered and applied in training too (round 1 later-work 6), or replaced by fit quality. |
| 4 | **Two-level (model × shot) bootstrap and the fold-ratio trap.** A mean of per-fold skill ratios is hostage to tail shots; the Nadeau–Bengio fold interval and the pooled out-of-fold readout give different `V_rot` readings; an uncorrected interval over overlapping splits is too narrow. 100 GPU runs over ten repeated 5-fold partitions exist. | §8bv–§8by, `docs/paper/b27c_hier.json`, `b27_kfold_interval_R10.json` | A (registered, 6/6 predictions held; registration order disclosed) | headline intervals | Statistics-of-evaluation material for an ML venue; the 100-run grid is a reusable testbed for any new readout without retraining. |
| 5 | **MNAR reweighting.** Reweighted onto genuinely missing rows: `T_i` vs persistence +0.28…+0.44 (8/8); vs PCHIP the interval excludes 0 on 2/4 cut and 4/4 inclusive splits. Only 54–68 % of missing `T_i` rows and 4–6 % of missing `V_rot` rows are in the W = 2 domain. Missingness rises with local activity (0.77 → 4.45 % across neighbour-change quintiles; §8bb P07). | §8i, §8ab.9, `paper_numbers.json → mnar` | B | one stress-test sentence | The in-domain fraction is the more important number: the scored task is a 10–20 ms-ahead prediction (Δt ≤ 15 ms for ≥ 96 % of scored rows). This is why the artificial-gap / multi-horizon curves of §3 are the highest-value new experiment. |
| 6 | **Reach / context saturation.** Trained at each reach, `T_i` saturates at 50 ms of contiguous context (deficit 20 ms −0.066, 30 ms −0.033, 40 ms −0.021); the operator family (LSTM, TCN, attention, SSM) does not decide skill at matched reach; width is flat over a 26× range; a 21k-parameter K = 8 latent matches the backbone in the cut population (+0.002). | §8ac, §8ae–§8am, §8z, §8aa, §8ag | A (dense ladder, pooled re-scoring) | one sentence | "The input set, not the model, is the ceiling" as a measured statement with four closed axes. Useful as the negative space of an ML paper; not a headline by itself. |
| 7 | **20-parameter surrogate (B.28).** Gap-dependent observation gain (K₀ ≈ 0.28, K∞ ≈ 0.51, s_K ≈ 42 ms) + linear response to each fast channel's deviation from its 100 ms baseline, cut off near 3σ; trained from scratch it keeps 0.72–0.78 of the backbone's PCHIP-relative skill (4/4 cut) and 0.31–0.50 of its causal-GP-relative skill; against the causal GP itself the interval excludes 0 on 2/4 cut splits. Weight-readout report on 30 frozen checkpoints: forget-gate memory 1.3 steps; effective contributions carried CES 47 %, BES 32 %, ECEI 12 %, MC 5 %. | §8bz–§8bz-4, `docs/b28_weight_physics_report.md`, `docs/paper/b28_*.json`, A2 | B (form chosen from TEST-row Jacobians: post hoc) | demoted to one post hoc sentence | A distillation-to-equation result. Loses the post hoc label if refit and scored on the B.27c partitions r2–r9, where the form was never selected. |
| 8 | **Multimodal coupling bound (A3).** Noise-corrected within-discharge correlation of fast-channel and target increments: `V_rot` \|r\| ≤ 0.044 over lags 0–200 ms and windows 10–200 ms; `T_i` lower bound ≥ +0.127 at lag 0 for every window. A torque-balance term audit and the Mirnov aliasing derivation back it. | `review_loop/analyses/A3`, `docs/paper/fast_target_coupling_bound.json`, §8ar, §8bd | A (model-free, all 641 files) | lead of Claim 2 | The cleanest physics statement in the repo. For NF it needs r/a and more than one chord to become a statement about momentum vs heat transport rather than about one location. |
| 9 | **Uncertainty / calibration: already evaluated three ways.** (i) Split conformal on the backbone: best Winkler score in 32/32 cells, coverage 0.87–0.92 at nominal 0.90 (§8m, §8ab.4). (ii) Heteroscedastic Gaussian head: fails calibration, rejected on `V_rot` (§8bf). (iii) Two-component mixture and Student-t heads: CRPS skill vs an oracle-scale Gaussian +0.22 (cut) / +0.43–0.44 (inclusive) for `T_i`, about +0.35 for `V_rot`, 16/16 splits; 50 % / 90 % coverage 0.48–0.55 / 0.90–0.92; fitted ν ≈ 2.1–3.5; the mechanism is heavy tails (zero bimodal rows) (§8bf, §8bj). | §8m, §8ab.4, §8bf, §8bj, `data/.b17b_summary.json` | A for marginal calibration in both populations (B.17b) | cut from the paper in round 2 (R2-5) | A probabilistic-nowcast section is close to free: the trained runs exist. Missing: conditional coverage by event stratum and by gap length, and reliability diagrams as figures. |
| 10 | **Campaign (temporal) split and per-shot standardization.** With per-discharge input z-scoring the backbone keeps +0.17…+0.20 vs PCHIP across a temporal split (4/4 inits, both populations); the cause of the unstandardized failure is measured (BES drifts 1.22σ, targets 0.115σ). | §8n, §8s, §8ab.6 | B (the standardization is offline, and it was adopted after seeing that split) | Limitations | With a causal (expanding-window) standardizer this becomes a covariate-shift result that survives review; see §3 item 1b. |
| 11 | **Per-discharge skill covariates.** Skill vs the causal GP correlates with the discharge-level spread of the truth (ρ = +0.40 `T_i`, +0.28 `V_rot`, Bonferroni p ≤ 0.0024); `T_i` wins 85 % of the most variable tercile, `V_rot` 55 %. Burst strata on four raw test shots: the gain over PCHIP sits 10–50 ms after a burst (+0.32 cut), while rows with a burst inside the causal 10 ms window show +0.01 and carry 51 % of the model SSE. | §8an, §8be, `data/.b6_burst_strata.json` | C (exploratory, no registered rule; four shots for the burst strata) | not used | The seed of event-conditioned evaluation. "No gain inside the burst, gain in the recovery" is physically interesting and rests on four discharges. |
| 12 | **Offline gap-filling benchmark with artificial gaps (B.12).** Observed rows hidden in spans drawn from the real missing-run-length distribution; every learned arm is below PCHIP at 1-step gaps (−0.26) and at +0.39…+0.63 for 2–10 and > 10 steps (cut `V_rot`, routed BiLSTM); with zero same-shot labels no arm is above the train mean; a MAST-pretrained transformer adds nothing over random initialization. | §8av, §8az, `data/.b12_masks/`, `data/.b12_card.json` | B (multi-axis card, registered) | not used | The masking infrastructure for the artificial-gap curve the reviewers asked for already exists in offline form. The causal form has not been run. |
| 13 | **Real-time cost.** Stateful step 0.97 ms median / 1.49 ms p99 on CPU; cost is dispatched operator count; the GPU is slower at this model size. | §8l, §8ac, §8ah–§8aj | B (absolutes are machine-dependent) | one paragraph | Supports a "deployable" framing only once a consumer is named and standardization is causal. |

**Never tried** (verified by searching `THESIS_RESULTS.md` and `PROJECT_KNOWLEDGE.md`):
multi-step / multi-horizon forecasting (the only mention is the argument that NBI matters at
100 ms and longer); a causal artificial-gap curve; memoryless MLP or gradient-boosting baselines
on the same 22 inputs; an expanding-window or EWMA input standardizer; a history-only retrain of
`seq_v2` (only evaluation-time zeroing of the windowed control exists); `V_rot` on 30–50 ms
averaged labels; a zero-shot pretrained time-series baseline; the four CPU items of round 2 §6
(a)–(d) (only A1–A3 exist under `review_loop/analyses/`).

---

## 3. Cheapest high-value experiments not yet run

**Measured cost on this machine** (`data/.b14_batch.log`, B.27 / B.27c totals): one `seq_v2`
run = training 2.2–7.5 min (25–30 epochs, 357,570 parameters) + test scoring 2.6–6 min
(CPU-bound: per-row causal GP refits and bootstrap) ≈ **7 min**. B.27 = 20 runs in 2.24 h;
B.27c = 80 runs in 9.60 h. A "4 splits × 2 populations" batch takes about **1 h**; with 4 inits,
about 4 h. The GPU was idle during this pass (`nvidia-smi`: 0 MiB in use). Project rule: a GPU
batch is never substituted by CPU training.

| # | experiment | new code | runs | wall time | what it decides |
|---|---|---|---|---|---|
| 1a | **Causal standardization, train-file statistics** (`CES_PER_SHOT_NORM=0` under `GATE_ENV`): 4 splits × 2 populations + temporal split (4 inits × 2) | none (knob exists: `seq/train_seq.py`, `seq/seq_data.py:207`) | 8 + 8 | ~2 h | Whether "causal" can return to the title. Old-protocol evidence: `T_i` vs PCHIP excludes 0 on 4/4 (`data/.seq_v2_nops_lstm_s*`). The temporal split is expected to degrade (§8n). |
| 1b | **Expanding-window (or EWMA) standardizer** | ~30 lines in `seq_data.py` + a test; the cache key changes | 8 + 8 | ~2 h + half a day of code | Whether the campaign-shift repair survives without future statistics. This is the deployable variant and the gate for any real-time claim. |
| 2 | **History-only `seq_v2` retrain** (fast channels zeroed in the `T_i` branch) | a zero-fast switch analogous to `CES_SEQ_ZERO_MC` (~5 lines) | 8 (32 with 4 inits) | 1 h (4 h) | The fast-channel share of the +0.113 over the causal GP. Round 2 ranks it equal first; it is the central attribution of Claim 1. |
| 3 | Round 2 §6 CPU items: (a) `V_rot` running-mean arm under the A1 rule; (b) surrogate with a_c = 0 (4-parameter K(s) filter) on frozen rows; (c) Table 1 `V_rot` column under the rule; (d) causal-GP arm attached to the `per_shot=0` artifacts | small scripts on existing npz | 0 | 4 × ≤ 25 min CPU | (b) is the cheapest measurement of the fast-channel share; (a) decides whether Claim 2 narrows to "CES-history smoother level". |
| 4 | **Causal artificial-gap curve**: hide the last k = 1, 2, 5, 10, 20 real labels before each scored row; all arms (backbone, surrogate, persistence, causal GP, EWMA; PCHIP / linear / offline GP with the matching two-sided mask) | scoring script; reuse the span sampler in `experiments/b12/b12_data.py`. Inference on frozen checkpoints is in-distribution only up to the staleness seen in training, so add one retrained arm with label dropout | 0 (inference) + 8 retrains | 2–3 h inference/scoring (causal GP dominates) + 1 h GPU | Turns a 10–20 ms-ahead result into a skill-vs-gap curve out to 200 ms. Prior from §8al and B.12: the fast-channel term matters more as the CES anchor ages. The experiment most likely to change what the paper can claim. |
| 5 | **Multi-horizon forecasting** (predict t + h, h = 10…200 ms, from inputs up to t): one multi-output head | target shift in `seq_data.py`; multi-head copy of `model_seq_v2.py` through the existing model registry | 8 | 1–2 h + about a day of code | Whether any fast channel leads the CES target. A3 puts the `T_i` coupling maximum at lag 0, so the prior is a decaying curve; a measured lead-time bound is still a result. NBI is argued to matter only at 100 ms and longer (§8bd addendum), and NBI is not in the data. |
| 6 | **Probabilistic output on the gap / horizon curves** | none for the heads (`v2t`, `v2m2` exist); per-stratum scoring in `b17_dist/score_b17b.py` | 0 on existing rows; 8–16 if combined with 4 or 5 | < 1 h CPU; 1–2 h GPU | Conditional calibration by gap length and activity stratum; reliability figures. Marginal calibration is already established (§8bj). |
| 7 | **Event-conditioned evaluation** | tagger scripts (burst detector exists in `b6_mus/`; BES-off and level-step proxies of §1.3) | 0 | ≤ 1 h CPU on frozen npz | Row-level strata on all 641 shots with input-activity proxies; named-event strata only on the raw shots (4 test shots in B.6; scoring through the B.27 folds puts every raw shot out-of-fold, up to 12). Exploratory until a reference signal arrives. |
| 8 | **Memoryless baselines** (MLP and gradient boosting on the same 22 inputs incl. carried CES + staleness; Diag2Diag-style) | ~100 lines; GBM on CPU, MLP on GPU | 8 + 8 | < 1 h | Fills the baseline gap named in both review rounds (base-1, H2-5). Prior: close to the 21k-parameter rung, since the model's memory is 1.3 steps (B.28 report). |
| 9 | AR(p) and Kalman/OU rows fitted on TRAIN for Table 1 | mostly exists (`b16_physics`, §8bi anchors) | 0–8 | < 1 h | Completes the CES-only causal comparator set with intervals (round 2 G7: these exist only as verbs). |
| 10 | **Surrogate refit on B.27c partitions r2–r9** | runner loop over existing fold manifests | 16 (two repeats) to 80 | 2–9 h | Removes the post hoc label from the 20-parameter filter. |
| 11 | `V_rot` on 30–50 ms averaged labels | label aggregation + rescoring; one retrain set | 0–8 | 1 h | Tests the rotation claim at a scale where the fit-noise floor is lower (round 1 later-work 8). |
| 12 | Direct routing ablation under the confirmed protocol (all 22 channels into the `V_rot` branch) | model variant through the registry | 8 | 1 h | Removes the "earlier protocol" tag from the routing evidence. |
| 13 | Zero-shot pretrained time-series model as a control | external weights; inference only | 0 | 1–2 h | The only item left open by the 2026-09-05 literature survey. Low priority for NF, moderate for an ML venue. |

Suggested order by value per hour: 3 → 2 → 1a → 4 → 8 → 1b → 6/7 → 5 → 10. Items 1a + 2 + 3 + 8
fit in one working day; adding 4 and 1b takes about three days including code and registration
(every batch needs its `PREREGISTRATION_*.md` committed before the first run).

---

## 4. What cannot be done without new data, and what to request

| blocked item | why blocked | request to the KSTAR team (exact) |
|---|---|---|
| Replace the 3 keV and 1,000 km/s value rules; verify the noise-floor reading | no fit quality in the files | Per-exposure CES fit output for channel 18 on the 641 shots: fit chi-square, fitted line amplitude / photon counts, 1σ fit errors on `T_i` and `V_rot`, fit-status flag. |
| Whether missing rows have an exposure (the NN-CES distinction); cause of the held `CES_VT18` values | no frame timing, no hold provenance | CES frame timestamps and exposure time, the beam-modulation log, the reason `CES_VT18` is forward-filled in 499 files. |
| Profile-level or transport-level physics (heat vs momentum, pedestal vs core) | one chord | `CES_TI` / `CES_VT` for all CES channels (the other radial chords), ideally with fit errors. |
| Radial location in flux coordinates, confinement regime | no equilibrium | EFIT time series: r/a (or ψ_N) of R = 2.198 m, q95, κ, δ, R_sep, W_MHD, β_N; plus I_p and line-averaged density. |
| Named-event evaluation (ELM, L-H, H-L); beam-state confounding | no reference signal | Divertor D-alpha at ≥ 1 kHz for all 641 shots; per-source NBI on/off, power and energy waveforms; ECH and RMP coil-current waveforms. |
| `V_rot` forecasting at 100 ms and longer; the torque term | the driver is not in the dataset | The NBI waveforms above (torque proxy = power × tangency geometry per source). |
| External validity (the reason both rounds put *Nuclear Fusion* at about 5–6 %) | one campaign | The same 18-column product for a second campaign (2023 or 2024), for one frozen-protocol test registered beforehand. |
| Event statistics beyond 12 shots; BES positive controls | raw streams for 12 shots only | Raw BES / ECEI (500 kHz) and Mirnov (2 MHz) for more shots, chosen by label density rather than by publication (published shots are `V_rot`-label-poor); clarification of `MC1T06` vs `MC1T16` and exact BES channel positions. |
| A consumer for the nowcast | not a data file | Whether real-time CES exists in the plasma control system and at what latency; which controller or analysis would read a 10 ms `T_i` estimate. |
| Absolute `T_e` from ECEI, density from BES | uncalibrated channels | ECEI calibration factors, or co-located Thomson / ECE radiometer data. |
| Public benchmark (needed for a datasets-style ML contribution) | the data are not ours to release | Written data-use and release terms from KFE; co-authorship or acknowledgment terms. |

---

## 5. Reading for the venue decision

- **Solid today:** the `T_i` margin over a CES-only causal GP (+0.113 [+0.085, +0.132] cut, eight
  new partitions), the model-free coupling bound (A3), calibrated heavy-tailed heads (B.17b), the
  tail-vs-median diagnosis, and a deep evaluation record (about 80 recorded sections, a 100-run
  repeated k-fold grid).
- **What limits a main venue:** the scored task is a 10–20 ms-ahead fill at one chord in one
  campaign; the fast-channel share of the margin is not isolated; input standardization is offline;
  there is no memoryless baseline; the architecture (two-branch LSTM) is not new; the data cannot
  be released.
- **What local compute can address in days:** attribution (items 2, 3b), causality (1a, 1b),
  horizon (4, 5), baselines (8, 9), conditional calibration (6), the surrogate's post hoc label (10).
- **What only new data can address:** a second campaign, more chords, fit quality, EFIT, D-alpha
  and NBI. These are the items both review rounds name for *Nuclear Fusion*.
