# B.40 — an empirical error bar for the CES value: a bracket for the white-noise variance, checked on MAST where fit errors are reported, and its dependence on the beam-emission level

Registered 2026-11-18, before any of the readouts below was computed. CPU only; no model. Designed by the session lead and executed by a delegated agent from this document.

## 1. Why

The quantity the paper estimates is the T_i value that the KSTAR CES analysis reports. The author's question (2026-11-18): how far is that from the true ion temperature, and what can be measured about it. The tenth independent grading put the same point first: the files carry no per-exposure fit error. The random part of (reported − true) can be estimated from the series itself; the systematic part cannot and is the subject of a separate literature check.

## 2. Seen before

The lag-1 autocorrelation of the 10 ms T_i increments on the chord (−0.416, manuscript). B.37 part Q: minus the lag-1 autocovariance of the increments on the test discharges gives 120.9 to 162.6 eV (plain) and 85.6 to 114.2 eV (increments clipped at their 1st and 99th percentiles) as a lower bound of the noise standard deviation. The MAST gate and edge studies (the share of the increment variance that the reported errors explain; the regime number by radius). Not seen: an upper bound, a robust version, any dependence of the bound on the beam-emission level or on the T_i level, the same estimators on MAST beside the reported errors.

## 3. Model and estimators

A measured value is y_t = s_t + e_t with e_t white of variance σ². With d_t = y_t − y_{t−1}:

- lower estimator N_lo = −cov(d_t, d_{t+1}) = σ² − cov(Δs_t, Δs_{t+1}), which is at most σ² when the increments of the signal are not negatively correlated;
- upper estimator N_hi = var(y_{t+1} − 2 y_t + y_{t−1}) / 6 = σ² + var(Δ²s)/6, which is at least σ².

Three versions of each, over triples of consecutive measured rows inside a segment: moments; moments with the differences clipped at their 1st and 99th percentiles; and robust, with S(x) = 1.4826 × the median absolute deviation: N_hi = S²(y_{t+1} − 2y_t + y_{t−1}) / 6 and N_lo = [S²(y_{t+1} − 2y_t + y_{t−1}) − S²(y_{t+1} − y_{t−1})] / 4. **The robust version is primary**; the other two are reported. Intervals: 95 %, 2,000 resamples of discharges (shots on MAST).

## 4. MAST: is the reported fit-error variance inside the bracket?

Shots, blocks and scored chords as in the MAST edge study (`mast/run_gate.py::prepare`, population G: relative T_i error below 0.1; population W, below 0.5, is reported). Normalized-radius bins 0–0.3, 0.3–0.5, 0.5–0.7, 0.7–0.9, 0.9–1.0 as in `mast/run_coupling.py`. Reference: the median over the same rows of the squared reported error (the mean is reported beside it).

- **M1.** In at least four of the five bins the reference lies between N_lo and N_hi (point values), and pooled over all bins the interval of N_lo over the reference lies below 1 and that of N_hi over the reference above 1.

## 5. KSTAR

All 641 files, the 10 ms grid (first and last row of a file dropped), T_i with held values and values above 3 keV removed (the cut population); the inclusive population is reported.

- **K1 (the bracket is tight).** Pooled over all discharges, N_hi / N_lo is at most 1.30 and the upper end of its interval at most 1.45.
- **K2 (the white part scales with the measurement signal).** Triples are divided into fifths by the BES level of the centre row (mean of the nine channels) relative to the median of its discharge. N_lo in the lowest fifth over N_lo in the highest fifth has an interval above 1.
- **Reported.** σ = √N_lo and √N_hi in eV overall, by fifth of the BES level, by third of the T_i level (mean of the three rows), and the ratio σ / level in each third; the distribution of the per-discharge σ (discharges with at least 100 triples); on the test rows of the four frozen row tables (`data/.b30/rows_cut_s*.npz`, natural cadence): the mean squared error of hold, the causal GP, the history-only twin and the causal variant divided by N_lo and by N_hi of the same discharges (how many noise floors each estimator sits at), and the largest skill over hold that an estimator of the signal could reach, 1 − N / MSE_hold.

## 6. What goes into the paper

- M1 and K1 hold: the bracket is stated as the random part of (reported − true T_i) on the chord, in eV and relative to the level, with the MAST check as its validation; scores are also given in units of the floor.
- K2 holds: the white component is described as scaling with the measurement signal, as fit noise does. K2 fails: it is described as a white component at 10 ms whose origin (fit noise or fluctuations faster than a frame) the files do not decide.
- K1 fails: only the lower bound is stated.

## 7. Expectations

KSTAR, robust: √N_lo 40 to 60 eV, N_hi / N_lo 1.05 to 1.25 (K1 holds). K2: ratio 1.3 to 3 (holds). σ / level 0.06 to 0.15, largest in the lowest third of the level. Causal variant at 2.5 to 5 floors of the robust bound in mean squared error. MAST: M1 holds in at least four bins; N_lo is negative in the innermost bins.

## 8. Limits known in advance

White noise and non-negatively correlated signal increments are assumptions; a real fluctuation faster than a frame is counted as noise. Heteroscedastic noise makes a robust scale a typical value, not a mean. The BES level stands in for the charge-exchange signal and also follows the plasma density. Nothing here bounds a bias of the reported value.
