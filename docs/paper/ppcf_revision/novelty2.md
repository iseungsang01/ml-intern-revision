# Simulated PPCF novelty referee report

*This is simulated feedback. It is not a journal decision.*

## Summary of claims (author framing kept)

The paper builds a causal nowcast of the reported C VI ion temperature on one KSTAR pedestal-top chord (R = 2.198 m). The estimator is a Matérn‑3/2 GP on past CES values followed by an untuned boosted regression on history scalars plus BES/ECEI scalars (§3.2, Table 2). It is compared with hold, a causal GP, PCHIP and an offline two-sided GP under four validation designs (§3.5). The main reported results are:

- **Against interpolation:** MSE is lower than PCHIP in all four designs. The weakest is chronological, at 15.3 [6.5, 22.7]% (Table 4).
- **Against the offline GP:** a reduction is established only under the random and cluster designs. The cluster figure is 3.7 [0.7, 7.0]%.
- **Value of the fast channels:** small with the previous frame visible. Larger with a stale history: 19.0 [3.3, 29.2]% under rolling origin at D650 (Table 6), and 27.7 [19.5, 35.8]% on MAST using Thomson at D100.
- **Runtime:** about 2 ms per row, p99 4.6 ms (§4.3, Table S13).

## Established components (not novel; the authors say so in §1)

- **Neural reading of raw CES spectra** goes back to JET. Bishop et al. framed the spectrum analysis as a nonlinear mapping learned by an MLP (https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/bishop-cxrs-ppcf-93.pdf). Svensson et al. built a demonstration real-time ion-temperature profile diagnostic on that approach (https://www.osti.gov/etdeweb/biblio/20059540).
- **Speed of spectrum readers:** EAST reported roughly 0.1 ms per frame, three orders of magnitude faster than the conventional fit (http://pst.hfcas.ac.cn/en/article/doi/10.1088/2058-6272/ab2674). A KSTAR CES neural-network analysis exists as well (the paper's ref [5]). So a ~2 ms latency is not distinctive in itself.
- **Real-time virtual kinetic profiles:** RTCAKENN is running in the DIII-D plasma control system with accuracy comparable to offline CAKE (https://www.osti.gov/pages/biblio/2251556-machine-learning-based-real-time-kinetic-profile-reconstruction-diii). A DIII-D extended Kalman filter estimate of Te is cited as ref [26].
- **Methods:** GP regression, gradient boosting, PCHIP, blocked/cluster bootstrap and TCNs are all standard (§1, "What is established").
- **Literature gap:** a third search on delayed/missing-sensor nowcasting outside fusion was not run. I make no novelty inference from that gap. The authors themselves concede a "wide literature" exists (§1).

## Application-specific evidence

- **The data:** a carefully audited 2022 data set (627–639 discharges). Fast samples are snapshots at the frame start, and absent frames are mapped to beam notches (§2.1–2.2, S1, Table S3).
- **Rigour:** registered batches with failed statements disclosed (Tables S33–S36), independent recomputation (S9), permutation and anchor-row controls (Table S17), and seed spread (Table S15).
- **Honest negatives:**
  - No difference from the offline GP is established in time order.
  - The offline GP has the lower median error (ratio 1.06 [1.02, 1.10], Table 4).
  - The second stage has to be refitted for each condition (Tables S9, S10).
  - The radius-specificity claim is withdrawn: Mirnov signals carry about a quarter of the increment (Table S19).
  - The last block of shots fails under rule S (Table 8).

## The supported distinct contribution

The defensible contribution is a quantified result for one chord. In the staleness/sparsity regime (D50–D1000, S, Kk), fast local channels recover roughly half of the MSE lost to history age under the random design (recovered share 0.51 at D650, Table 5). That share is smaller under the stricter designs (0.28 rolling origin, 0.46–0.49 chronological; Table S7). The direction is replicated on MAST with a different diagnostic.

The nowcast-condition result, with the previous frame visible, is weaker. Beating PCHIP is robust. Beating the offline GP is design-dependent and concentrated in the tails (90th-percentile ratio 0.95).

One structural point limits the significance. The beam-modulation schedule S is an assumed reading with zero analysis time (§2.2), not a measured operational latency. And the paper's own ref [5] (KSTAR NN spectrum reader) and ref [6] (two-Gaussian fit) aim to remove the delay at its source. Since the paper does not rank against them (§5), the practical importance of the staleness regime is open.

## TCN evidence (judged separately)

These are budgeted fits, so they say nothing about architectural ordering at convergence (§4.7).

- **Full suite, one initialization (Table 10):**
  - The fast-input TCN is worse than the history TCN at D10 under V2, V3 and V4a (for example V3 −0.518 [−0.717, −0.348]).
  - Positive skill is established only under V1 (D650, S, K10).
  - Its RMSE values (198–290 eV) are far above the GP/tree estimator's.
- **Fixed subset, three initializations (Table 9):**
  - Positive at V1 D650/S and V2 S only.
  - Negative at V3 and V4a D10.

These two scopes must not be pooled, and the authors correctly do not pool them. The TCN section adds evidence that the GP+trees choice is reasonable. It adds no contribution of its own, and is somewhat orthogonal to the five questions.

## Classification of issues

- **Rewrite only:**
  - Clarify that S is hypothetical.
  - Shorten §4.7. Its generated-table prose reads as audit output; also note that Table 11's model labels (TH/TX) are opposite to the main text's TX/T0 naming.
  - Compress the hedging so that the abstract states the staleness result first, without changing the framing.
- **New analysis on existing fits:**
  - A per-shot-cluster breakdown of the offline-GP comparison.
  - A test of whether the gain over hold and PCHIP is concentrated in ELM or notch frames (data in Tables S5/S18).
- **New fits (CPU is enough, no GPU needed):** fair-budget or converged TCN runs are optional and should not be required.
- **New measurements/data (decisive, not obtainable by rewriting):**
  - A second campaign that has never been read.
  - Measured CES analysis latency.
  - The output of a real-time spectrum reader on the same discharges.
  - Per-shot field and ECEI settings, to explain the failure of the last block.

## Ranked changes, with falsifiers

1. **Score the estimator on an untouched campaign** (2023/2024) under V4-style fitting. *Falsifier:* the D650 or S fast-channel increment interval includes zero, or the gain over PCHIP vanishes.
2. **Ground the staleness regime in operational reality.** Either use the measured KSTAR CES delivery latency, or run the ref [5] reader on the same shots. *Falsifier:* if the reader delivers each frame within one frame time, the D650/S regime has no operational relevance and only the small D10 increment remains.
3. **Settle the last-block failure** using the toroidal field and ECEI local-oscillator metadata. *Falsifier:* if the failure tracks ECEI resonance moving off the chord, the gain depends on diagnostic configuration rather than being robust.

## Recommendation: **weak reject**

The analysis is rigorous and unusually candid. But the methodological novelty is nil by the authors' own account, and the empirical claim rests on one chord, one campaign that had already been read, and a hypothetical latency schedule. Against the strongest reference, the two-sided GP, the advantage disappears in time order.

**Can rewriting alone get this to weak reject or better?** It is already at weak reject. Rewriting alone could plausibly hold it there, or move it to borderline if the hypothetical status of S and the limited scope of the contribution are stated plainly. Rewriting cannot reach weak accept or higher. That would need change 1 or change 2, which require new data or measurements that are not in the current files.