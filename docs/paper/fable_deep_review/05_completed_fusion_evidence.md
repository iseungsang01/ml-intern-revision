# Source audit: RTCAKENN, COMPASS and Diag2Diag versus the KSTAR pedestal-top CES nowcast

**Conventions.** Tags: [R] read directly, [A] author claim, [H] my application hypothesis, [U] not established.
- RTCAKENN and COMPASS locators are PDF page numbers of the supplied files (PDF page = printed page + 1).
- Diag2Diag was supplied as EuropePMC body text only, so locators are section headings. Its figures, supplement and peer-review file were not supplied.
- Diag2Diag code is `PlasmaControl-Diag2Diag-SRTS-7e938a3/diag2diag_natcomm_model_train.py`. I cite function and statement; line numbers cannot be counted reliably from the escaped string. The shot list it loads (`diag2diag_shotlist.txt`) was not supplied.

## 1. RTCAKENN (Shousha et al., Nucl. Fusion 64 026006)

1. **Input and target.** [R] Inputs are Bt, Ip, 15 MSE pitch angles, 8 boundary points, RT-EFIT p and q, TS ne/Te, and CER Ti/Ωtor on a 33-point ψN grid. Targets are seven offline CAKE profiles on 121 points, including Ti (Table 1, Fig. 1, PDF p.4). "Ion" quantities are C6+, and CAKE excludes edge C6+ temperatures and extrapolates via a Ti/Te assumption (§2, PDF p.3).
2. **Target diagnostic as input.** [R] Yes, the current (most recent valid) CER Ti is an input. There is no history: the authors call it a model with no memory (§4.2.2, PDF p.13).
3. **Split and order.** [R] 696 discharges and 18,959 time points; the test set is 1,334 time slices not used in training, with early stopping on validation loss (§2, PDF p.5). [U] Whether the split is by discharge or by slice, and its temporal order, are not stated. [R] PCS runs on #195928 and #196114–196117 are separate experimental demonstrations (§4, PDF pp.7–12); [U] their relation to the training database is not stated.
4. **High-rate validation.** [R] None. Comparisons are with offline CAKE at four slices 500 ms apart (Fig. 3, PDF p.7) and at one slice (Fig. 5, PDF p.9). The real-time alternative there came from post-shot playback (PDF p.8).
5. **Missing and stale sensors.** [R] Three mechanisms are described:
   - Training applies input dropout to TS or CER at rate 0.1, replacing values with zeros or random in-range values (PDF p.5).
   - At run time each input uses its most recent valid sample, with outlier and flag checks; ring buffers were rejected, and indefinite staleness is acknowledged (§3, PDF p.6).
   - The effect of absence duration is deferred to future work; the only numbers are ≈15% versus ≈13% profile discrepancies at two slices (PDF p.14).
6. **Computation versus deployment.** [R] It was deployed in the DIII-D PCS. The whole function (input reception, ten networks, averaging) took ≈13 ms in the first version and ≈7.7 ms later, on a 20 ms cycle (§4.3, PDF pp.14–15; Fig. 16, PDF p.17). [R] Control use is stated as future (abstract, PDF p.2). [U] Diagnostic acquisition and transfer latency is not reported.
7. **Closest overlap.** [H] It is a deployed real-time carbon-impurity Ti output built on last-valid CER hold, analogous to our hold baseline plus cross-diagnostic correction. It has no temporal nowcast, no anchor-age input and no fast fluctuation diagnostics.
8. **Material uncertainty.** [U] Split unit. [R] Ti and Vtor have the weakest regression (R≈0.937/0.939, Fig. 2, PDF p.6), and the authors flag unexplained Ti variance (PDF p.17).

## 2. COMPASS (Imrisek et al., PPCF 68 065049)

1. **Input and target.** [R] Inputs are AXUV (2×20 channels) and SXR (17), plus interferometer, Dα/C III, 16 IPR coils, Ip, Uloop, coil supplies and position (§2.1, PDF pp.4–5; Table 1, PDF p.5). Targets are TS Te/ne profile vectors. TS runs at ≤120 Hz; inputs are sampled at up to 2 MHz (PDF p.4).
2. **Target diagnostic as input.** [R] No. Kinetic profiles are targets only, explicitly contrasted with RTCAKENN (§1, PDF p.3). It is a per-sample MLP with no history (§2.3, PDF pp.6–7).
3. **Split and order.** [R] 2,764 discharges and 51,476 profiles, split 80/20 **by discharge**, plus five-fold cross-validation inside the training set (§2.2, PDF pp.5–6). [R] Architecture and checkpoint were selected on validation performance (PDF pp.6–7), so there is no selection-free test set. [U] Chronological order and whether z-score statistics were train-only are not stated.
4. **High-rate validation.** [R] None directly. Errors are evaluated at TS times (Figs. 7–10). The fast-event evidence is one validation discharge, #21772, with sparse TS points and qualitative consistency with SXR/Dα (§3.4, Fig. 18, PDF pp.15–16). The authors expect larger uncertainty during ELMs because they are under-represented (PDF p.16).
5. **Missing and stale sensors.** [R] Not tested. Fig. 12 (PDF p.12) retrains with one diagnostic added to the base model; it is not a missingness test. Data were restricted to a period of stable geometry and settings (PDF p.5). Raw versus 50 µs-smoothed inputs differ modestly (core Te MAPE 5.40 versus 4.72%, Table 3, PDF p.14). [U] Whether the smoothing is causal.
6. **Computation versus deployment.** [R] ≈0.1 ms inference per profile; real-time use is described only as potential (PDF p.17). No deployment.
7. **Closest overlap.** [H] It is the nearest analogue of our BES/ECEI-only control: fast indirect signals mapped to a slow kinetic target with no target anchor. Its edge errors (≈20%) exceed core errors (≈10%) (PDF p.17). It covers electrons only.
8. **Material uncertainty.** [A] Impurity-driven non-uniqueness of the radiation-to-profile map (PDF p.17). [H] Raw BES/ECEI voltages may have analogous unobserved gain and setting confounders.

## 3. Diag2Diag (Jalalvand et al., Nat. Commun. 16 8506)

1. **Input and target.** [R] 236 inputs: CER 58, interferometer 4, ECE 40, magnetics 8, MSE 38, plus first and second derivatives of ECE and interferometer. Targets are 80 TS channels (Methods, "Feature extraction"). Code `extract_features` agrees: X stacks `co2_density`, `ece_cali` and their derivatives, `cer_ti`, `mse`, `magnetics`; Y stacks `ts_core_density` and `ts_core_temperature`.
2. **Target diagnostic as input.** [R] No TS input (Introduction, "Contribution"; the X list in `extract_features`). CER Ti is input only, never a target.
3. **Split and order.** [R] Paper: 4,000 randomly selected discharges from 2017–2022, ≈3.4M points split 80/20; the unit is not stated in the text. [R] Code: `train_discharges = 2000`, then `X_data[:train_discharges]` versus the remainder, which is a **discharge-level split in shot-list order**. [U] Two limits on that reading:
   - The published `generate_datasets` uses random dummy arrays (the real loader is commented out), so the production split is not verified.
   - Whether the showcase discharges (153761, 174823, 157545, 136219) were excluded from training is not stated; the script loads 174823 as its example.

   [R] Hyperparameters were searched over 2,000 iterations on the validation set.
4. **High-rate validation.** [R] This is the strongest of the three, but partial:
   - Burst-mode TS (up to 10 kHz, 3–7 pulses) on #153761, with validation R²=0.92 at TS times.
   - An aggregate of more than 20 ELM cycles on #174823.
   - Consistency with simulation for the RMP cases.

   The authors state that no MHz ground truth exists (Methods, "Validation methodology and limitations").
5. **Missing and stale sensors.** [R] All inputs are aligned to their most recent sample; CER (200 Hz) and MSE are held constant between samples (Methods; code `reindex(ref_time, method='ffill')`). [A] Robustness to target-diagnostic failure is claimed. [U] No input-dropout or staleness test.
6. **Computation versus deployment.** [R] Offline analysis only; no inference time or PCS deployment is reported.
7. **Closest overlap.** [H] It is the closest to our "fast diagnostic increment": stale CER is forward-filled alongside fast ECE and fluctuation channels to infer a slow diagnostic between its samples. The target is TS, not Ti; there is no anchor or age input; alignment is at target times.
8. **Material uncertainty.** [R] Two properties of the published script, with [U] whether either reflects the production pipeline:
   - `mycollate_fn` zero-pads discharges, and `nn.L1Loss` is applied without a mask.
   - In `extract_features`, the error bands are built from normalized value ± normalized error. Because the error was also mean-subtracted, the bands become (y−e)/s and (y+e−2m)/s, not symmetric about (y−m)/s.

## 4. Corrections to earlier claims

- **"No discharge-level holdout in this literature" is wrong.** COMPASS states a by-discharge split (PDF p.6). The Diag2Diag script splits by discharge index. Only RTCAKENN leaves the unit unstated. None documents a chronological or campaign holdout or a selection-free test set, so our lack of an untouched campaign is a shared limitation, not a unique one.
- **"No deployed inference across a diagnostic gap" is wrong, with qualifications.** RTCAKENN ran in the PCS during plasma discharge #195928 without valid TS input, caused by a ψN bug (§4.2.1, PDF pp.10–12). The no-CER case was a hardware test, #914928, that simulated the absence (PDF pp.12–15). Both were whole-discharge absences judged qualitatively. No source quantifies error versus staleness between reports of the target diagnostic in real time. COMPASS and Diag2Diag gap inference is offline.
- [H] Neither correction shows that pedestal-top CES nowcasting is anticipated or unanticipated; three sources cannot settle prior art.

## 5. Three hypotheses testable with our data

All three would be TEST-informed and exploratory, registered before any fit or readout. Source-age rules remain non-horizons, and if a component is already covered by existing runs it should be dropped rather than refit.

**H1. Anchor-corruption training (RTCAKENN-motivated).**
- Hypothesis: one model trained with random CES-anchor dropout or age inflation shows a fast increment over causal GP that grows monotonically with anchor age under the existing stale/sparse schedules.
- Readout: paired discharge-cluster differences against the existing conditional-availability fits.
- Falsifier: an age-flat or negative interaction.
- Caveats: a null is not equivalence. This is distinct from B104, which varies budget, not availability.

**H2. Report noise versus plasma signal (COMPASS/Diag2Diag-motivated).**
- Hypothesis: if BES/ECEI carry state information and not report-specific noise, the paired fast increment scored against L5 is at least as large as against F on identical samples.
- Readout: a registered paired contrast, not overlap of marginal intervals.
- Falsifier: an increment present only for F suggests fitting report idiosyncrasy.
- Caveats: L5 is carbon-impurity, possibly MNAR, and has its own smoothing bias, so this discriminates but does not establish truth.

**H3. Sub-10 ms information (Diag2Diag derivatives, COMPASS smoothing).**
- Hypothesis: in the small high-rate subset only, strictly past-window features (≈1 ms mean, first and second differences) add to the 10 ms raw snapshots for the same learner.
- Caveats: the subset is underpowered, so report effect size with cluster interval and no generalization. A positive result indicates snapshot aliasing; it does not identify turbulence physics or calibrated ne/Te.
