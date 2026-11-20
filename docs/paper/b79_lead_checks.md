# B.79 — the session lead's checks

Written 2026-11-20 after the batch was read, with code written independently of the implementing agent's
(`ces_prediction/experiments/b79_beam_inputs/lead_checks/verify_b79.py`). Skill of A over B = 1 - SSE(A) / SSE(B) on identical rows; target L5.

`py ces_prediction/experiments/b79_beam_inputs/lead_checks/verify_b79.py .`

```
(A) own pooling of the stored estimates (main readout), target L5
  N1 TN over T0, V1, D650 +0.148 | N2 TXN over TN, V1, D650 +0.258 | N3 V2 +0.298 | N4 V3 +0.263 | N5 V4a +0.245
  N6 TXN over TN, V1, S +0.253 | N7 V4a, S +0.102 | N8 TN over T0, V4a, S -0.100 | N9 TXN over TX, V1, D650 +0.047 | N10 TXN over TN, V1, D10 +0.097
  TN over T0, D650: V2 -0.044, V3 -0.126, V4a -0.066 | TXN over TX, D650: V2 -0.020, V3 -0.025, V4a -0.054
(B) own beam scalars under D650
  largest stamp difference between a beam row and its grid row: 9.5e-07 s; files with an invalid power column: 13 [30801, 30805, 30809, 30810, 30811, 30812, 30842, 30843, 30845, 30895, 30906, 30907, 30909]
  cached beam scalars: 19 columns; for each, the smallest largest-deviation to one of my 19 columns: max over columns 0.00e+00 (rows of the common set: 160343; all finite in mine: True)
(B, C) own fits, V4a, D650, L5
  columns as delivered: fit rows 127270, test rows 10146; RMSE T0 202.3, TN 208.9, TX 179.1, TXN 178.3; TN over T0 -0.066, TXN over TN +0.271, TX over T0 +0.217, TXN over TX +0.008
     stored: RMSE T0 202.3, TN 208.9, TX 176.7, TXN 181.4; TXN over TN +0.245; largest |mine - stored| T0 0.00e+00, TN 0.00e+00 eV
  without the 13 files: fit rows 125487, test rows 10146; RMSE T0 203.0, TN 207.6, TX 182.0, TXN 180.4; TN over T0 -0.045, TXN over TN +0.245, TX over T0 +0.196, TXN over TX +0.018
```

Pooled again from the two sets of stored estimates (the main readout in `data/.b79/fits/`, the refit in `data/.b79/refit_valid/fits/`):

```
main (as delivered): scored rows of one V1 re-partition, D650: 160343
   TXN over TN, D650: V1 +0.258, V2 +0.298, V3 +0.263, V4a +0.245 | S: V1 +0.253, V4a +0.102 | TN over T0, D650, V1 +0.148; S, V4a -0.100 | TXN over TX, D650, V1 +0.047
refit without the 13 files: scored rows of one V1 re-partition, D650: 158560
   TXN over TN, D650: V1 +0.248, V2 +0.299, V3 +0.246, V4a +0.249 | S: V1 +0.250, V4a -0.070 | TN over T0, D650, V1 +0.149; S, V4a +0.091 | TXN over TX, D650, V1 +0.040
```

- A: the values of N1 to N10 of the main readout equal those of `b79_beam_inputs.md`; so do the refit's.
- B: the nineteen beam scalars rebuilt from the NBI files with own code (join by shot and stamp) equal the cached ones exactly on the common set; the validity rule selects the thirteen files and no other. The lead's own fits of T0 and TN on the chronological design equal the stored estimates bit for bit; TX and TXN use the lead's own fast scalars, which differ from the agent's at rounding level, and give +0.271 for TXN over TN where the stored fits give +0.245 (registered interval [+0.061, +0.357]).
- C: the lead's own refit of that unit without the thirteen files gives +0.245 for TXN over TN.

Order of events, as it happened: the agent reported the thirteen files with invalid NBI-2 power columns before any fit and named a default (columns as delivered, the registration read literally). The lead's instruction to leave the files out was sent before the fits but reached the agent only after the fits and the readout. The registered statements are therefore judged on the readout with the columns as delivered, and the refit without the files, run after that readout had been seen, is reported beside it and judged on nothing.
